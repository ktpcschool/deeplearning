"""
Interface 2018年12月号（CQ出版社）
特集2 AIひょっこり猫カメラ
のコードを改変。

カラスを発見したら画像を保存、
鷹の音声ファイルを再生。
"""

import collections
import cv2
from datetime import datetime
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.preprocessing import image
import math
import multiprocessing
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import sys
import time
import traceback

from word_tree import WordTree


class Sound(multiprocessing.Process):
    def __init__(self, mp3_file):
        self.mp3_file = mp3_file
        super().__init__()

    def run(self):
        # 音声ファイルの読み込み
        sound = AudioSegment.from_file(self.mp3_file, 'mp3')

        # 再生
        play(sound)

        # 20秒間停止
        time.sleep(20)


# 画像判定時、上位何個までを判定対象とするか。
# 1つの画像に複数の物が写っていた場合、上位からこの数の分だけ対象の物なのか判定する。
PREDICT_NUM = 10

# カメラの解像度
CAMERA_RESOLUTION = (400, 300)
# 部分画像で切り取って判定する際の縦横の長さ
CROP_LEN = int(CAMERA_RESOLUTION[1] * 0.5)
# 部分画像どうしが重なる幅
MARGIN = int(CROP_LEN * 0.5)
# 部分画像を切り取る際にずらしていく幅
STEP_LEN = CROP_LEN - MARGIN
# 画像判定のバッチ数
BATCH = 1


class DetectionCamera(object):
    def __init__(self, target_name):
        self.target_name = target_name
        self.word_tree = WordTree()

        # ターゲット名が、ImageNetの単語ツリー内に存在するか検索。
        # 存在すれば、その単語配下の全単語について単語IDを検索し、保持する。
        # 存在しなければプログラム終了。
        word_id = self.word_tree.find_id(self.target_name)
        if word_id is None:
            print('{0} is not ImageNet word. Retry input.'.format(
                self.target_name))
            exit(0)

        self.target_ids = self.word_tree.list_descendants(word_id)

        # 画像判定モデルを初期化。
        print('Start model loading...')
        self.model = MobileNet(weights='imagenet')

    def match_target_image(self, frame):
        """
        引数の画像に何が写っているか判定し、検知対象であればその物体名を返す。
        :param frame: 確認したい画像
        :return: 物体名　検知対象でなければNone
        """

        # 画像の内容を配列化
        img_list = []
        org_img = image.array_to_img(frame, scale=True)
        img_list.append(org_img)

        # タイル状に切り取り
        # 小さく写っている物体も検知できるよう、元画像を小さくタイル状に切り取った画像群を作る。
        # タイルは縦横がCROP_LENの正方形とする。
        # 境目に物体があるときも対応できるようMARGIN分の幅で重なりが出るように切り取る。
        org_width, org_height = org_img.size
        crop_left_list = self._split_step(org_width)
        crop_top_list = self._split_step(org_height)

        for left in crop_left_list:
            for top in crop_top_list:
                right = left + CROP_LEN
                bottom = top + CROP_LEN
                tile_img = org_img.crop((left, top, right, bottom))
                img_list.append(tile_img)

        # 全画像を判定
        # 元画像＋タイル画像群の全てについて、画像判定モデルを用いて、目標の物体が写っているか判定する。
        preds_list = self._predict_image(img_list, top=PREDICT_NUM)

        # 判定結果のチェック
        # 全画像分の判定結果について、写っている物体が検知したい対象かをチェックする。
        # 検知対象の物体であれば、その名前を記録する。
        pred_name_list = []

        for index, preds in enumerate(preds_list):
            for pred_id, pred_name, score in preds:
                if pred_id in self.target_ids:
                    pred_name_list.append(pred_name)

        # ランキングTopの取得
        # 記録した名前でランキングを作り、1位のものの名前を返す。
        pred_name_ranking = collections.Counter(pred_name_list)

        top_name_count = pred_name_ranking.most_common(1)
        if len(top_name_count) == 0:
            return None
        else:
            ret, _ = top_name_count[0]
            return ret

    def _split_step(self, target):
        """
        引数targetの長さを、CROP_LENずつ、MARGIN分の幅で重なりが出るように区切ったときの
        始点の座標を返す
        :param target: 区切る対象の長さ
        :return: 座標(int)の配列
        """
        end_point = target - CROP_LEN
        step_num = int(math.floor(end_point / STEP_LEN))
        ret = [point * STEP_LEN for point in range(step_num + 1)]
        ret.append(end_point)
        return ret

    def _predict_image(self, img_list, top=10):
        """
        画像判定モデルを用いて、何が写っているか判定する。
        :param img_list: 判定画像のリスト
        :param top: 上位何位まで判定するか
        :return: 検知結果　(単語ID, 名称, 判定スコア)のタプルがtopで指定された数のリストになる。
        """
        # 判定画像を配列化する。
        img_array_list = []
        for img in img_list:
            resized = img.resize((224, 224))
            img_array = image.img_to_array(resized)
            img_array = preprocess_input(img_array)
            img_array_list.append(img_array)

        # バッチサイズ分ずつ取り出し、判定にかける。
        ret = []
        for index in range(0, len(img_array_list), BATCH):
            x = np.array(img_array_list[index:index + BATCH])
            preds = self.model.predict(x)
            decoded = decode_predictions(preds, top=top)
            ret.extend(decoded)

        return ret


def main():
    try:
        # カメラを開始
        cap = cv2.VideoCapture(0)
        target_name = 'Corvine bird'
        camera = DetectionCamera(target_name)

        # カメラ画像を確認
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            # 目的のものが写っているか判定。
            matched_target = camera.match_target_image(frame)

            if matched_target is not None:
                print(matched_target)

                # 音声ファイルの再生
                mp3_file = 'hawk1.mp3'
                process1 = Sound(mp3_file)
                process1.start()

                # 画面を保存
                now = datetime.now()
                f = now.strftime('%Y-%m-%d-%H-%M-%S') + ".jpg"
                cv2.imwrite(f, frame)
                print("save=", f)

                process1.join()

                time.sleep(5)
                
            # 画面を表示
            cv2.imshow("frame", frame)

            # Escキーで終了
            if cv2.waitKey(5)&0xff == 27:
                break

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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
