import cv2
import numpy as np
import sklearn
import config
from utils import face_preprocess
from PIL import ImageFont, ImageDraw, Image
from utils.utils import feature_compare, load_mtcnn, load_faces, load_mobilefacenet, add_faces
import gradio as gr


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


# 人脸识别
def face_recognition(image):
    # 将 gradio 的 Image 组件接收的图像转换为 cv2 格式
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces, landmarks = mtcnn_detector.detect(frame)
    if faces.shape[0]!= 0:
        faces_sum = 0
        for i, face in enumerate(faces):
            if round(faces[i, 4], 6) > 0.95:
                faces_sum += 1
        if faces_sum == 0:
            return image
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
        for i, embedding in enumerate(embeddings):
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


# 创建识别人脸的界面
recognition_interface = gr.Interface(
    fn=face_recognition,
    inputs=gr.Image(streaming=True, type="numpy", label="实时摄像头"),
    outputs=gr.Image(label="识别结果"),
    title="人脸识别",
    description="请使用实时摄像头进行人脸图像识别"
)


if __name__ == '__main__':
    # 启动识别人脸的界面
    recognition_interface.launch()