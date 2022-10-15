import os

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import time
import cv2
import moviepy.editor
from moviepy.editor import *
import torch


pts, left_side, right_side = pickle.load(open('gold_points.pkl', 'rb'))
def get_size(bbox):
    line = int((bbox[1] * 2 + bbox[3]) // 2)
    return max(bbox[2], bbox[3]) / (right_side[line] - left_side[line]) * 1600


def get_class(size):
    ranges = [(250, 1600), (150, 250), (100, 150), (80, 100), (70, 80), (40, 70), (0, 40)]
    for i in range(7):
        if size > ranges[i][0] and size <= ranges[i][1]:
            return i + 1


@st.cache
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    print('model loaded')

    return model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')


def process_frame(path, image, model):
    results = model(path)

    preds = []
    for result in results.xyxy[0]:
        xmin, ymin, xmax, ymax, confidence = result[0], result[1], result[2], result[3], result[4]
        preds.append([xmin, ymin, xmax, ymax])

    preds = torch.as_tensor(preds)

    count = len(preds)
    sizes = [get_size([x[0], x[1], x[2]-x[0], x[3]-x[1]]) for x in preds.numpy()]
    sizes = [x for x in sizes if x <= 1600]
    classes = [get_class(size) for size in sizes]

    for i, pred in enumerate(preds):
        pred = [round(x) for x in pred.numpy()]
        try:
            if classes[i] in [1]:
                cv2.rectangle(image, pt1=(pred[0], pred[1]), pt2=(pred[2], pred[3]), color=(255, 0, 0), thickness=2)
            else:
                cv2.rectangle(image, pt1=(pred[0], pred[1]), pt2=(pred[2], pred[3]), color=(0,255,0), thickness=2)
        except:
            pass

    return image, sizes, classes, count


def load_write_process(vid, rate=1/30, frameName='frame'):
    vidcap = cv2.VideoCapture(vid)
    clip = moviepy.editor.VideoFileClip(vid)

    model = load_model()

    seconds = clip.duration
    print('durration: ' + str(seconds))

    count = 0
    frame = 0

    clips = []
    success = True
    logs_to_csv = []
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, frame * 1000)
        success, image = vidcap.read()

        ## Stop when last frame is identified
        # print(frame)
        if frame > seconds or not success:
            break
    # for i in range(len(os.listdir('test/'))):
        print('extracting frame ' + str(frame) + '-%d.png' % count)

        os.makedirs('test/', exist_ok=True)
        path = f'test/test_frame{frame}.jpg'
        # print(path)
        if image is not None:
            image = cv2.resize(image, (1280, 720))

            cv2.imwrite(path, image)
            # image = cv2.imread(path)
            image, sizes, classes, count_obj = process_frame(path, image, model)
            logs_to_csv.append([frame, sizes, classes, count_obj])

            clips.append(ImageClip(image).set_duration(rate))
            frame += rate
            count += 1

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile("test_infered.mp4", fps=1/rate)

    df = pd.DataFrame(logs_to_csv, columns=['frame', 'sizes', 'classes', 'count_obj'])
    df['mean_size'] = df['sizes'].apply(lambda x: np.mean(x))
    df['count_for_class'] = df['classes'].apply(lambda x: [x.count(i) for i in range(1, 8)])
    df['min_size'] = df['sizes'].apply(min)
    df['max_size'] = df['sizes'].apply(max)
    return df


@st.cache
def loading(name):
    return load_write_process(name)


st.set_page_config(
    page_title="Real-Time поиск негабаритов",
    page_icon="🤖",
    layout="wide",
)

title_col, warn_col = st.columns(2)
title_col.title('Baby Kagglers')

kpi1, kpi2 = st.columns(2)
video = kpi1.file_uploader("Выберите видео для загрузки", type=["mp4", "avi"])

if video:
    start_time = time.time()
    df = loading(video.name)

    print('time:', time.time()-start_time)

    kpi1.video('test_infered.mp4')
    clip = moviepy.editor.VideoFileClip('test_infered.mp4')
    seconds = clip.duration

    ths = kpi2.selectbox('Выберите порог негабарита (больше порога - негабарит)', options=[
        '1 - более 250 мм',
        '2 - до 250 мм',
        '3 - до 150 мм',
        '4 - до 100 мм',
        '5 - до 80 мм',
        '6 - до 70 мм',
        '7 - до 40 мм',
    ])

    ths = int(ths[0])

    if ths != 1:

        time.sleep(1)

        kpi2.markdown("### Визуализация тренда изменения грансостава")
        placeholder_kpi2 = kpi2.empty()

        placeholder_warn_col = warn_col.empty()

        placeholder = st.empty()
        change = 0

        i = 0

        for frames_len in range(0, len(df), 10):  ############ костыль

            if frames_len < 370:
                sec_frames = df.iloc[:frames_len+1]['frame']
                mean_sizes = df.iloc[:frames_len+1]['mean_size']

                max_sizes = df.iloc[:frames_len+1]['max_size']
                min_sizes = df.iloc[:frames_len+1]['min_size']
            else:
                sec_frames = df.iloc[175+i:frames_len+1]['frame']
                mean_sizes = df.iloc[175+i:frames_len+1]['mean_size']

                max_sizes = df.iloc[175+i:frames_len+1]['max_size']
                min_sizes = df.iloc[175+i:frames_len+1]['min_size']

                i += 10

            fig = px.line(x=sec_frames, y=mean_sizes, labels={
                         "x": "Секунда",
                         "y": "Среднее значение крупности руды",
                     },)
            fig.update_layout(height=500)
            placeholder_kpi2.plotly_chart(fig)

            with placeholder.container():
                warn_er, met1, met2, met3 = st.columns(4, gap='large')

                past_count = df.iloc[frames_len - 1]['count_obj'] if frames_len >0 else 0
                past_mean = df.iloc[frames_len - 1]['mean_size'] if frames_len > 0 else 0
                past_max = max(df.iloc[frames_len - 1]['sizes']) if frames_len > 0 else 0

                met1.metric(
                    label="Кол-во камней, шт",
                    value=round(df.iloc[frames_len]['count_obj']),
                    delta=round(df.iloc[frames_len]['count_obj'] - past_count)
                )

                met2.metric(
                    label="Среднее значение крупности камней, мм",
                    value=round(df.iloc[frames_len]['mean_size']),
                    delta=round(df.iloc[frames_len]['mean_size'] - past_mean)
                )

                met3.metric(
                    label="Максимальный размер камня, мм",
                    value=round(max(df.iloc[frames_len]['sizes'])),
                    delta=round(max(df.iloc[frames_len]['sizes']) - past_max)
                )

                gabs = [i for i in df.iloc[frames_len]['classes'] if i < ths]
                if change == 0:
                    warn_er.error(
                        f'Обнаружен негабариты классов {np.unique(gabs)}. Количество негабаритов: {len(gabs)}')
                    change = 1
                elif change == 1:
                    warn_er.warning(
                        f'Обнаружен негабариты классов {np.unique(gabs)}. Количество негабаритов: {len(gabs)}')
                    change = 0

                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.markdown('### Мониторинг размера самого большого/маленького камня')
                    t = pd.DataFrame({'max_size': max_sizes, 'min_size': min_sizes, 'sec_frames': sec_frames})
                    fig3 = px.line(t, x='sec_frames', y=['max_size', 'min_size'], labels={
                        "x": "Секунда",
                        "value": "Размер",
                        'variable': 'size'
                    }, )
                    st.plotly_chart(fig3)


                with fig_col2:
                    st.markdown("### Мониторинг рудного потока по классам крупности")
                    fig2 = px.bar(y=df.iloc[frames_len]['count_for_class'], x=np.arange(1, 8).astype(str), labels={
                         "x": "Класс",
                         "y": "Количество камней",
                     })
                    fig2.update_layout(yaxis_range=[0, 12])

                    st.plotly_chart(fig2)

            time.sleep(1/4.5)














