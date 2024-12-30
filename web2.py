import cv2
import numpy as np
import gradio as gr
import config
from utils import face_preprocess
from PIL import ImageFont, ImageDraw, Image
from utils.utils import feature_compare, load_mtcnn, load_faces, load_mobilefacenet, add_faces
import sklearn


# 人脸识别阈值
VERIFICATION_THRESHOLD = config.VERIFICATION_THRESHOLD

# 检测人脸检测模型
mtcnn_detector = load_mtcnn()
# 加载人脸识别模型
face_sess, inputs_placeholder, embeddings = load_mobilefacenet()
# 添加人脸
add_faces(mtcnn_detector)
# 加载已经注册的人脸
faces_db = load_faces(face_sess, inputs_placeholder, embeddings)


# 注册人脸
def face_register():
    print("点击y确认拍照！")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('y'):
            faces, landmarks = mtcnn_detector.detect(frame)
            if faces.shape[0]!= 0:
                faces_sum = 0
                bbox = []
                points = []
                for i, face in enumerate(faces):
                    if round(faces[i, 4], 6) > 0.95:
                        bbox = faces[i, 0:4]
                        points = landmarks[i, :].reshape((5, 2))
                        faces_sum += 1
                if faces_sum == 1:
                    nimg = face_preprocess.preprocess(frame, bbox, points, image_size='112,112')
                    user_name = input("请输入注册名：")
                    cv2.imencode('.png', nimg)[1].tofile('face_db/%s.png' % user_name)
                    print("注册成功！")
                else:
                    print('注册图片有错，图片中有且只有一个人脸')
            else:
                print('注册图片有错，图片中有且只有一个人脸')


# 人脸识别
def face_recognition(frame):
    # 检查 frame 是否为 None
    if frame is None:
        return None
    faces, landmarks = mtcnn_detector.detect(frame)
    if faces.shape[0]!= 0:
        faces_sum = 0
        for i, face in enumerate(faces):
            if round(faces[i, 4], 6) > 0.95:
                faces_sum += 1
        if faces_sum == 0:
            return frame
        # 人脸信息
        info_location = np.zeros(faces_sum)
        info_name = []
        probs = []
        # 提取图像中的人脸
        input_images = np.zeros((faces.shape[0], 112, 112, 3))
        for i, face in enumerate(faces):
            if round(faces[i, 4], 6) > 0.95:
                bbox = faces[i, 0:4]
                points = landmarks[i, :].reshape((5, 2))
                nimg = face_preprocess.preprocess(frame, bbox, points, image_size='112,112')
                nimg = nimg - 127.5
                nimg = nimg * 0.0078125
                input_images[i, :] = nimg

        # 进行人脸识别
        feed_dict = {inputs_placeholder: input_images}
        emb_arrays = face_sess.run(embeddings, feed_dict=feed_dict)
        emb_arrays = sklearn.preprocessing.normalize(emb_arrays)
        for i, embedding in enumerate(emb_arrays):
            embedding = embedding.flatten()
            temp_dict = {}
            # 比较已经存在的人脸数据库
            for com_face in faces_db:
                ret, sim = feature_compare(embedding, com_face["feature"], 0.70)
                temp_dict[com_face["name"]] = sim
            dict = sorted(temp_dict.items(), key=lambda d: d[1], reverse=True)
            if dict[0][1] > VERIFICATION_THRESHOLD:
                name = dict[0][0]
                probs.append(dict[0][1])
                info_name.append(name)
            else:
                probs.append(dict[0][1])
                info_name.append("unknown")

        for k in range(faces_sum):
            # 写上人脸信息
            x1, y1, x2, y2 = faces[k][0], faces[k][1], faces[k][2], faces[k][3]
            x1 = max(int(x1), 0)
            y1 = max(int(y1), 0)
            x2 = min(int(x2), frame.shape[1])
            y2 = min(int(y2), frame.shape[0])
            prob = '%.2f' % probs[k]
            label = "{}, {}".format(info_name[k], prob)
            cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)
            font = ImageFont.truetype('font/simfang.ttf', 18, encoding="utf-8")
            draw.text((x1, y1 - 18), label, (255, 0, 0), font=font)
            frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame


def main():
    with gr.Blocks() as demo:
        with gr.Tab("注册人脸"):
            register_button = gr.Button("注册人脸")
            register_button.click(face_register)
        with gr.Tab("识别人脸"):
            video_input = gr.Video()
            output = gr.Image()
            start_button = gr.Button("开始识别")
            stop_button = gr.Button("停止识别")
            start_button.click(face_recognition, inputs=[video_input], outputs=[output])
            stop_button.click(lambda: None)

    # 设置 share=True 以创建公共链接
    demo.launch(share=True)


if __name__ == '__main__':
    main()