"""
Copyright (c) 2013 Daniel Bader (http://dbader.org)
License: MIT
https://github.com/dbader/schedule

Copyright(c) 2019 Tatsuro Watanabe
License: MIT
https://github.com/ktpcschool/deeplearning
"""
import cv2
from datetime import datetime
import glob
import logging
import numpy as np
import os
import shutil
import sys

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import schedule


# 基本ディレクトリ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ハンドラを生成する
std_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(BASE_DIR, 'security_camera.log'))

# フォーマット、ログレベル、ハンドラを設定する
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)s - %(message)s',
                    level=logging.ERROR,
                    handlers=[std_handler, file_handler])

logger = logging.getLogger(__name__)


def make_video_from_image(image_path, video_dir, fps, size):
    """
    画像ファイルから動画を作成
    :param image_path: 画像ファイルのパス
    :param video_dir: 動画ファイルのパス
    :param fps: フレームレート
    :param size: 画像ファイルのサイズ
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    count = 1
    name = '{}/out{}.avi'.format(video_dir, count)
    video = cv2.VideoWriter(name, fourcc, fps, size)

    for i, filename in enumerate(sorted(glob.glob(image_path))):
        img = cv2.imread(filename)
        video.write(img)

        # 画像ファイル3000枚ごとにビデオ作成
        if i > 0 and i % 3000 == 0:
            count += 1
            name = '{}/out{}.avi'.format(video_dir, count)
            video = cv2.VideoWriter(name, fourcc, fps, size)

    video.release()


def upload_to_google_drive(file, mime_type):
    """
    googleドライブにファイルをアップロードする
    :param file: アップロードするファイル
    :param mime_type: アップロードするファイルのMIMEタイプ
    """
    gauth = GoogleAuth()
    gauth.CommandLineAuth()
    drive = GoogleDrive(gauth)
    folder_id = os.environ['FOLDER_ID']  # フォルダーIDは環境変数から取得
    f = drive.CreateFile({'title': file,
                          'mimeType': mime_type,
                          'parents': [{'kind': 'drive#fileLink',
                                       'id': folder_id}]})
    f.SetContentFile(file)
    f.Upload()


def upload_files_to_google_drive(path, mime_type):
    """
    指定のファイルをgoogleドライブにアップロードする
    :param path: アップロードするファイルのパス
    :param mime_type: アップロードするファイルのMIMEタイプ
    """
    for filename in glob.glob(path):
        upload_to_google_drive(filename, mime_type)


def delete_files(path):
    """
    指定したpathのファイルを削除
    :param path: 削除するファイルのパス
    """
    for file in glob.glob(path):
        os.remove(file)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # モジュール読み込み
    sys.path.append('/opt/intel/openvino/python/python3.5/armv7l')
    from openvino.inference_engine import IENetwork, IEPlugin

    # ターゲットデバイスの指定
    plugin = IEPlugin(device="MYRIAD")

    # モデルの読み込み
    model_path = 'models/person-detection-retail-0013.xml'
    weight_path = 'models/person-detection-retail-0013.bin'
    net = IENetwork(model=model_path, weights=weight_path)
    exec_net = plugin.load(network=net)

    # カメラ準備
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS
    # print(fps)

    probability_threshold = 0.5  # バウンディングボックス表示の閾値
    display_size = (400, 300)  # 表示するサイズ
    model_size = (544, 320)  # モデルが要求するサイズ
    transpose = (2, 0, 1)  # HWC → CHW（モデルによって変わる）
    image_path = 'video_image/*.jpg'  # 画像ファイルのパス
    video_dir = 'videos'  # 動画ファイルがあるディレクトリ
    video_path = '{}/out*.avi'.format(video_dir)  # 動画ファイルのパス
    mime_type = 'video/x-msvideo'

    schedule.every().day.at("19:00").do(make_video_from_image,
                                        image_path=image_path,
                                        video_dir=video_dir,
                                        fps=fps,
                                        size=display_size)
    schedule.every().day.at("19:30").do(upload_files_to_google_drive,
                                        path=video_path,
                                        mime_type=mime_type)
    schedule.every().day.at("20:00").do(delete_files,
                                        path=image_path)
    schedule.every().day.at("20:10").do(delete_files,
                                        path=video_path)

    try:
        # メインループ
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # 何らかのキーが押されたら終了
            if cv2.waitKey(1) != -1:
                break

            # 表示サイズに変換
            display_image = cv2.resize(frame, display_size)

            # 入力データフォーマットへ変換
            img = cv2.resize(frame, model_size)
            img = img.transpose(transpose)
            img = np.expand_dims(img, axis=0)

            # 推論実行
            out = exec_net.infer(inputs={'data': img})

            # 出力から必要なデータのみ取り出し
            detection_out = out['detection_out']
            detections = np.squeeze(detection_out)  # サイズ1の次元を全て削除

            # 現在時刻を表示
            now = datetime.now()
            now_sec = now.strftime('%m%d%H%M%S')
            cv2.putText(display_image, now_sec,
                        (display_size[0] // 2 + 50, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # 検出されたすべての領域に対して１つずつ処理
            for detection in detections:
                confidence = float(detection[2])

                # バウンディングボックス座標を入力画像のスケールに変換
                xmin = int(detection[3] * display_size[0])
                ymin = int(detection[4] * display_size[1])
                xmax = int(detection[5] * display_size[0])
                ymax = int(detection[6] * display_size[1])

                # confidence > 閾値 → バウンディングボックス表示
                if confidence > probability_threshold:
                    cv2.rectangle(display_image, (xmin, ymin), (xmax, ymax),
                                  color=(240, 180, 0), thickness=1)

                    # 画像ファイルに書き込む
                    now_mil = now.strftime('%m%d%H%M%S%f')
                    image_file = now_mil + ".jpg"
                    f = os.path.join(image_path, image_file)
                    cv2.imwrite(f, display_image)

            # 画像表示
            cv2.imshow('window', display_image)

            schedule.run_pending()

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error('エラー\nエラー内容:{}\nエラータイプ:{}\nファイル名:{}\n行番号:{}'.format(e, exc_type, fname, exc_tb.tb_lineno))
    finally:
        # カメラの終了処理
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
