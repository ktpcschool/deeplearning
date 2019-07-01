"""
Copyright(c) 2017 Intel Corporation.
License: MIT
https://github.com/movidius/ncappzoo

Copyright(c) 2018 JellyWare Inc.
License: MIT
https://github.com/electricbaka/movidius-ncs

Copyright(c) 2019 Tatsuro Watanabe
License: MIT
https://github.com/ktpcschool/deeplearning
"""

import cv2
from mvnc import mvncapi as mvnc
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import sys
import traceback


class Sound(object):
    def __init__(self, mp3_files):
        self.mp3_files = mp3_files

    def play_sound(self):
        for mp3_file in self.mp3_files:
            # 音声ファイルの読み込み
            sound = AudioSegment.from_file(mp3_file, 'mp3')

            # 再生
            play(sound)


# Neural Compute Stickのクラス
class NCS(object):
    def __init__(self, graph_path, mean_file, dim, threshold, max_detection):
        self.graph_path = graph_path
        self.mean_file = mean_file
        self.dim = dim
        self.threshold = threshold
        self.max_detection = max_detection

    # Device準備
    def open_ncs_device(self):
        # Get a list of available device identifiers
        devices = mvnc.enumerate_devices()
        if len(devices) == 0:
            print("No devices found")
            quit()

        # Initialize a Device
        device = mvnc.Device(devices[0])

        # Initialize the device and open communication
        device.open()

        return device

    # Graph準備と割り当て
    def load_graph(self, device):
        # Load graph file data
        with open(self.graph_path, mode='rb') as f:
            blob = f.read()

        # Initialize a Graph object
        graph = mvnc.Graph('graph')

        # Allocate the graph to the device
        fifo_in, fifo_out = graph.allocate_with_fifos(device, blob)

        return graph, fifo_in, fifo_out

    # 入力画像準備
    def load_image(self, frame):
        ilsvrc_mean = np.load(self.mean_file).mean(1).mean(1)
        img = frame
        img = cv2.resize(img, self.dim)
        img = img.astype(np.float32)
        img[:, :, 0] = (img[:, :, 0] - ilsvrc_mean[0])
        img[:, :, 1] = (img[:, :, 1] - ilsvrc_mean[1])
        img[:, :, 2] = (img[:, :, 2] - ilsvrc_mean[2])

        return img

    # 推論実行
    def infer_image(self, fifo_in, fifo_out, graph, img, labels):
        # Send the image to the NCS
        graph.queue_inference_with_fifo_elem(
            fifo_in, fifo_out, img, 'user object')

        # Get the result from the NCS
        output, _ = fifo_out.read_elem()

        # Get the indexes sorted by probability
        probability_indexes = output.argsort()[::-1][:self.max_detection + 1]
        print('\n------- predictions --------')

        label = None
        for i in probability_indexes:
            if output[i] > self.threshold:
                label = labels[i].split()[1]
                print('probability:' + str(output[i]) + ' is ' + label)

        return fifo_in, fifo_out, label

    # 後片付け
    def clean_up(self, fifo_in, fifo_out, device, graph):
        fifo_in.destroy()
        fifo_out.destroy()
        graph.destroy()
        device.close()


def main():
    try:
        labels_file = 'data/ilsvrc12/synset_words.txt'
        labels = np.loadtxt(labels_file, str, delimiter='\t')
        graph_path = 'caffe/GoogLeNet/graph'
        mean_file = 'data/ilsvrc12/ilsvrc_2012_mean.npy'
        dim = (224, 224)
        threshold = 0.05  # 確率のしきい値
        max_detection = 10  # 最大検出数
        ncs = NCS(graph_path, mean_file, dim, threshold, max_detection)
        device = ncs.open_ncs_device()
        graph, fifo_in, fifo_out = ncs.load_graph(device)

        # チェックする生物のリスト
        target_name_list = \
            ['magpie', 'jay', 'tabby',
             'tiger cat', 'Persian cat',
             'Siamese cat', 'Egyptian cat']

        # 音声ファイル
        mp3_files = ['hawk1.mp3']

        # カメラを開始
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            # 画面を表示
            cv2.imshow('image', frame)

            # Escキーで終了
            if cv2.waitKey(5) & 0xff == 27:
                break

            img = ncs.load_image(frame)
            fifo_in, fifo_out, label = \
                ncs.infer_image(fifo_in, fifo_out, graph, img, labels)
            if label in target_name_list:
                sound = Sound(mp3_files)
                sound.play_sound()

    except Exception as ex:
        # エラーの情報をsysモジュールから取得
        info = sys.exc_info()

        # tracebackモジュールのformat_tbメソッドで特定の書式に変換
        tbinfo = traceback.format_tb(info[2])

        # 収集した情報を読みやすいように整形して出力する
        print('Python Error.'.ljust(30, '='))
        for tbi in tbinfo:
            print(tbi)
        print('  %s' % str(info[1]))
        print('\n'.rjust(30, '='))

    finally:
        # カメラの終了処理
        cap.release()
        cv2.destroyAllWindows()

        # 終了処理
        ncs.clean_up(fifo_in, fifo_out, device, graph)


if __name__ == '__main__':
    main()
