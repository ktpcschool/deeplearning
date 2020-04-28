"""
Copyright(c) 2017 Intel Corporation.
License: MIT
https://github.com/movidius/ncappzoo

Copyright(c) 2018 JellyWare Inc.
License: MIT
https://github.com/electricbaka/movidius-ncs

Copyright (c) 2013 Daniel Bader (http://dbader.org)
License: MIT
https://github.com/dbader/schedule

Copyright(c) 2019 Tatsuro Watanabe
License: MIT
https://github.com/ktpcschool/deeplearning
"""
import cv2
import glob
import numpy as np
import os
import schedule
import shutil
import sys
import traceback

from datetime import datetime
from mvnc import mvncapi as mvnc
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


# Neural Compute Stickのクラス
class NCS(object):
    def __init__(self, graph_path):
        self.graph_path = graph_path

    # Device準備
    @staticmethod
    def open_ncs_device():
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
    @staticmethod
    def load_image(img):
        img = img.astype(np.float32)
        img = np.divide(img, 255.0)
        img = img[:, :, ::-1]

        return img

    # 推論実行
    @staticmethod
    def infer_image(fifo_in, fifo_out, graph, img):
        # Send the image to the NCS
        graph.queue_inference_with_fifo_elem(
            fifo_in, fifo_out, img, 'user object')

        # Get the result from the NCS
        output, _ = fifo_out.read_elem()

        return output

    # Interpret the output from a single inference of TinyYolo (GetResult)
    # and filter out objects/boxes with low probabilities.
    # output is the array of floats returned from the API GetResult but converted
    # to float32 format.
    # input_image_width is the width of the input image
    # input_image_height is the height of the input image
    # Returns a list of lists. each of the inner lists represent one found object and contain
    # the following 6 values:
    #    string that is network classification ie 'cat', or 'chair' etc
    #    float value for box center X pixel location within source image
    #    float value for box center Y pixel location within source image
    #    float value for box width in pixels within source image
    #    float value for box height in pixels within source image
    #    float value that is the probability for the network classification.
    def filter_objects(self, inference_result, input_image_width, input_image_height, labels, probability_threshold):

        # tiny yolo v1 was trained using a 7x7 grid and 2 anchor boxes per grid box with
        # 20 detection classes
        # the 20 classes this network was trained on

        num_classes = len(labels)  # should be 20

        grid_size = 7  # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
        anchor_boxes_per_grid_cell = 2  # the number of anchor boxes returned for each grid cell
        num_coordinates = 4  # number of coordinates

        # -------------------- Inference result preprocessing --------------------
        # Split the Inference result into 3 arrays: class_probabilities, box_confidence_scores, box_coordinates
        # then Reshape them into the appropriate shapes.

        # -- Splitting up Inference result

        # Class probabilities:
        # 7x7 = 49 grid cells.
        # 49 grid cells x 20 classes per grid cell = 980 total class probabilities
        class_probabilities = inference_result[0:980]

        # Box confidence scores: 7x7 = 49 grid cells. "how likely the box contains an object"
        # 49 grid cells x 2 boxes per grid cell = 98 box scales
        box_confidence_scores = inference_result[980:1078]

        # Box coordinates for all boxes
        # 98 boxes * 4 box coordinates each = 392
        box_coordinates = inference_result[1078:]

        # -- Reshaping

        # These values are the class probabilities for each grid
        # Reshape the probabilities to 7x7x20 (980 total values)
        class_probabilities = np.reshape(class_probabilities, (grid_size, grid_size, num_classes))

        # These values are how likely each box contains an object
        # Reshape the box confidence scores to 7x7x2 (98 total values)
        box_confidence_scores = np.reshape(box_confidence_scores, (grid_size, grid_size, anchor_boxes_per_grid_cell))

        # These values are the box coordinates for each box
        # Reshape the boxes coordinates to 7x7x2x4 (392 total values)
        box_coordinates = np.reshape(box_coordinates,
                                     (grid_size, grid_size, anchor_boxes_per_grid_cell, num_coordinates))

        # -------------------- Scale the box coordinates to the input image size --------------------
        self.boxes_to_pixel_units(box_coordinates, input_image_width, input_image_height, grid_size)

        # -------------------- Calculate class confidence scores --------------------
        # Find the class confidence scores for each grid.
        # This is done by multiplying the class probabilities by the box confidence scores
        # Shape of class confidence scores: 7x7x2x20 (1960 values)
        class_confidence_scores = np.zeros((grid_size, grid_size, anchor_boxes_per_grid_cell, num_classes))
        for box_index in range(anchor_boxes_per_grid_cell):  # loop over boxes
            for class_index in range(num_classes):  # loop over classifications
                class_confidence_scores[:, :, box_index, class_index] = np.multiply(
                    class_probabilities[:, :, class_index], box_confidence_scores[:, :, box_index])

        # -------------------- Filter object scores/coordinates/indexes >= threshold --------------------
        # Find all scores that are larger than or equal to the threshold using a mask.
        # Array of 1960 bools: True if >= threshold. otherwise False.
        score_threshold_mask = np.array(class_confidence_scores >= probability_threshold, dtype='bool')
        # Using the array of bools, filter all scores >= threshold
        filtered_scores = class_confidence_scores[score_threshold_mask]

        # Get tuple of arrays of indexes from the bool array that have a >= score than the threshold
        # These tuple of array indexes will help to filter out our box coordinates and class indexes
        # tuple 0 and 1 are the coordinates of the 7x7 grid (values = 0-6)
        # tuple 2 is the anchor box index (values = 0-1)
        # tuple 3 is the class indexes (labels) (values = 0-19)
        box_threshold_mask = np.nonzero(score_threshold_mask)

        # Use those indexes to find the coordinates for box confidence scores >= than the threshold
        filtered_box_coordinates = box_coordinates[box_threshold_mask[0], box_threshold_mask[1], box_threshold_mask[2]]

        # Use those indexes to find the class indexes that have a score >= threshold
        filtered_class_indexes = np.argmax(class_confidence_scores, axis=3)[
            box_threshold_mask[0], box_threshold_mask[1], box_threshold_mask[2]]

        # -------------------- Sort the filtered scores/coordinates/indexes --------------------
        # Sort the indexes from highest score to lowest
        # and then use those indexes to sort box coordinates, scores, class indexes
        sort_by_highest_score = np.array(np.argsort(filtered_scores))[::-1]
        # Sort the box coordinates, scores, and class indexes to match
        filtered_box_coordinates = filtered_box_coordinates[sort_by_highest_score]
        filtered_scores = filtered_scores[sort_by_highest_score]
        filtered_class_indexes = filtered_class_indexes[sort_by_highest_score]

        # -------------------- Filter out duplicates --------------------
        # Get mask for boxes that seem to be the same object by calculating iou (intersection over union)
        # these will filter out duplicate objects
        duplicate_box_mask = self.get_duplicate_box_mask(filtered_box_coordinates)
        # Update the boxes, probabilities and classifications removing duplicates.
        filtered_box_coordinates = filtered_box_coordinates[duplicate_box_mask]
        filtered_scores = filtered_scores[duplicate_box_mask]
        filtered_class_indexes = filtered_class_indexes[duplicate_box_mask]

        # -------------------- Gather the results --------------------
        # Set up list and return class labels, coordinates and scores
        filtered_results = []
        for object_index in range(len(filtered_box_coordinates)):
            filtered_results.append([
                labels[filtered_class_indexes[object_index]],  # label of the object
                filtered_box_coordinates[object_index][0],  # xmin (before image scaling)
                filtered_box_coordinates[object_index][1],  # ymin (before image scaling)
                filtered_box_coordinates[object_index][2],  # width (before image scaling)
                filtered_box_coordinates[object_index][3],  # height (before image scaling)
                filtered_scores[object_index]  # object score
            ])

        return filtered_results

    # Converts the boxes in box list to pixel units
    # assumes box_list is the output from the box output from
    # the tiny yolo network and is [grid_size x grid_size x 2 x 4].
    @staticmethod
    def boxes_to_pixel_units(box_list, image_width, image_height, grid_size):

        # number of boxes per grid cell
        boxes_per_cell = 2

        # setup some offset values to map boxes to pixels
        # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
        box_offset = np.transpose(
            np.reshape(np.array([np.arange(grid_size)] * (grid_size * 2)), (boxes_per_cell, grid_size, grid_size)),
            (1, 2, 0))

        # adjust the box center
        box_list[:, :, :, 0] += box_offset
        box_list[:, :, :, 1] += np.transpose(box_offset, (1, 0, 2))
        box_list[:, :, :, 0:2] = box_list[:, :, :, 0:2] / (grid_size * 1.0)

        # adjust the lengths and widths
        box_list[:, :, :, 2] = np.multiply(box_list[:, :, :, 2], box_list[:, :, :, 2])
        box_list[:, :, :, 3] = np.multiply(box_list[:, :, :, 3], box_list[:, :, :, 3])

        # scale the boxes to the image size in pixels
        box_list[:, :, :, 0] *= image_width
        box_list[:, :, :, 1] *= image_height
        box_list[:, :, :, 2] *= image_width
        box_list[:, :, :, 3] *= image_height

    # creates a mask to remove duplicate objects (boxes) and their related probabilities and classifications
    # that should be considered the same object.  This is determined by how similar the boxes are
    # based on the intersection-over-union metric.
    # box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
    def get_duplicate_box_mask(self, box_list):
        # The intersection-over-union threshold to use when determining duplicates.
        # objects/boxes found that are over this threshold will be
        # considered the same object
        max_iou = 0.10  # 0.35 > 0.10

        box_mask = np.ones(len(box_list))

        for i in range(len(box_list)):
            if box_mask[i] == 0:
                continue
            for j in range(i + 1, len(box_list)):
                if self.get_intersection_over_union(box_list[i], box_list[j]) > max_iou:
                    box_mask[j] = 0.0

        filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
        return filter_iou_mask

    # Evaluate the intersection-over-union for two boxes
    # The intersection-over-union metric determines how close
    # two boxes are to being the same box.  The closer the boxes
    # are to being the same, the closer the metric will be to 1.0
    # box_1 and box_2 are arrays of 4 numbers which are the (x, y)
    # points that define the center of the box and the length and width of
    # the box.
    # Returns the intersection-over-union (between 0.0 and 1.0)
    # for the two boxes specified.
    @staticmethod
    def get_intersection_over_union(box_1, box_2):

        # one diminsion of the intersecting box
        intersection_dim_1 = min(box_1[0] + 0.5 * box_1[2], box_2[0] + 0.5 * box_2[2]) - \
                             max(box_1[0] - 0.5 * box_1[2], box_2[0] - 0.5 * box_2[2])

        # the other dimension of the intersecting box
        intersection_dim_2 = min(box_1[1] + 0.5 * box_1[3], box_2[1] + 0.5 * box_2[3]) - \
                             max(box_1[1] - 0.5 * box_1[3], box_2[1] - 0.5 * box_2[3])

        if intersection_dim_1 < 0 or intersection_dim_2 < 0:
            # no intersection area
            intersection_area = 0
        else:
            # intersection area is product of intersection dimensions
            intersection_area = intersection_dim_1 * intersection_dim_2

        # calculate the union area which is the area of each box added
        # and then we need to subtract out the intersection area since
        # it is counted twice (by definition it is in each box)
        union_area = box_1[2] * box_1[3] + box_2[2] * box_2[3] - intersection_area;

        # now we can return the intersection over union
        iou = intersection_area / union_area

        return iou

    # GUIでのオブジェクトの表示
    @staticmethod
    def display_objects_in_gui(source_image, filtered_objects, size):
        # copy image so we can draw on it. Could just draw directly on source image if not concerned about that.
        display_image = source_image  # not copy
        source_image_width = source_image.shape[1]
        source_image_height = source_image.shape[0]

        x_ratio = float(source_image_width) / size[0]
        y_ratio = float(source_image_height) / size[1]

        # loop through each box and draw it on the image along with a classification label
        print('Found this many objects in the image: ' + str(len(filtered_objects)))
        for obj_index in range(len(filtered_objects)):
            center_x = int(filtered_objects[obj_index][1] * x_ratio)
            center_y = int(filtered_objects[obj_index][2] * y_ratio)
            half_width = int(filtered_objects[obj_index][3] * x_ratio) // 2
            half_height = int(filtered_objects[obj_index][4] * y_ratio) // 2

            # calculate box (left, top) and (right, bottom) coordinates
            box_left = max(center_x - half_width, 0)
            box_top = max(center_y - half_height, 0)
            box_right = min(center_x + half_width, source_image_width)
            box_bottom = min(center_y + half_height, source_image_height)

            print('box at index ' + str(obj_index) + ' is... left: ' + str(box_left) + ', top: ' + str(
                box_top) + ', right: ' + str(box_right) + ', bottom: ' + str(box_bottom))

            # draw the rectangle on the image.  This is hopefully around the object
            box_color = (0, 255, 0)  # green box
            box_thickness = 2
            cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

            # draw the classification label string just above and to the left of the rectangle
            label_background_color = (70, 120, 70)  # greyish green background for text
            label_text_color = (255, 255, 255)  # white text
            cv2.rectangle(display_image, (box_left, box_top - 20), (box_right, box_top), label_background_color, -1)
            # now = datetime.now()
            # now_str = now.strftime('%m%d%H%M%S')
            cv2.putText(display_image, filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5],
                        (box_left + 5, box_top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # 後片付け
    @staticmethod
    def clean_up(fifo_in, fifo_out, device, graph):
        fifo_in.destroy()
        fifo_out.destroy()
        graph.destroy()
        device.close()


def make_video_from_image(path, size):
    """
    画像ファイルから動画を作成
    :param path: 画像ファイルのパス
    :param size: 画像ファイルのサイズ
    """
    image_path = path + '/*.jpg'
    name = 'out.mp4'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(name, fourcc, 20.0, size)

    for filename in sorted(glob.glob(image_path)):
        img = cv2.imread(filename)
        img = cv2.resize(img, size)
        video.write(img)

    video.release()

    # 画像ファイルの全削除
    delete_all_files(path)


def delete_all_files(path):
    """
    指定したフォルダを削除後、再作成
    :param path: 削除するフォルダ
    """
    shutil.rmtree(path)
    os.mkdir(path)


def upload_to_google_drive(video_file):
    """
    googleドライブに動画ファイルをアップロードする
    :param video_file: アップロードする動画ファイル
    """
    gauth = GoogleAuth()
    gauth.CommandLineAuth()
    drive = GoogleDrive(gauth)
    folder_id = os.environ['FOLDER_ID']  # フォルダーIDは環境変数から取得
    f = drive.CreateFile({'title': video_file,
                          'mimeType': 'video/mp4',
                          'parents': [{'kind': 'drive#fileLink',
                                       'id': folder_id}]})
    f.SetContentFile(video_file)
    f.Upload()


def main():
    try:
        folder = os.path.dirname(__file__)
        os.chdir(folder)
        graph_path = 'tiny_yolo_graph'

        # 分類するクラス
        labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                  'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

        # Tiny Yolo assumes input images are these dimensions.
        size = (448, 448)

        display_size = (400, 300)   # 表示するサイズ

        # only keep boxes with probabilities greater than this
        probability_threshold = 0.25

        target_name_list = ('car', 'cat', 'person')
        ncs = NCS(graph_path)
        device = ncs.open_ncs_device()
        graph, fifo_in, fifo_out = ncs.load_graph(device)

        cap = cv2.VideoCapture(0)

        path = 'video_image'    # 画像ファイルのパス
        schedule.every().day.at("20:00").do(make_video_from_image,
                                            path=path,
                                            size=display_size)

        video_file = 'out.mp4'
        schedule.every().day.at("20:01").do(upload_to_google_drive,
                                            video_file=video_file)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Escキーで終了
            if cv2.waitKey(5) & 0xff == 27:
                break

            img = frame
            display_image = cv2.resize(img, display_size)
            img = cv2.resize(img, size, cv2.INTER_LINEAR)

            img = ncs.load_image(img)
            output = ncs.infer_image(fifo_in, fifo_out, graph, img.astype(np.float32))
            filtered_objs = ncs.filter_objects(
                output.astype(np.float32), img.shape[1], img.shape[0], labels, probability_threshold)
            print(filtered_objs)

            ncs.display_objects_in_gui(display_image, filtered_objs, size)

            # 現在時刻を表示
            now = datetime.now()
            now_sec = now.strftime('%m%d%H%M%S')
            cv2.putText(display_image, now_sec,
                        (display_size[0] // 2 + 50, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # 表示する
            cv2.imshow('window', display_image)

            # target_name_list内の物体が見つかれば画像に書き込む
            for filtered_obj in filtered_objs:
                if filtered_obj[0] in target_name_list:
                    # 画像ファイルに書き込む
                    now_mil = now.strftime('%m%d%H%M%S%f')
                    image_file = now_mil + ".jpg"
                    f = os.path.join(path, image_file)
                    cv2.imwrite(f, display_image)

            schedule.run_pending()

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
