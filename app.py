import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Simulador de Visão Computacional", layout="wide")

st.title("Simulação de Visão Computacional")
st.write("Detecção simples baseada em regras usando OpenCV.")

uploaded_file = st.file_uploader(
    "Escolha uma imagem",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        ratio = w / h

        if h > w * 1.5:
            label = "Pessoa"

        elif ratio > 1.5:
            label = "Carro"

        else:
            label = "Animal"

        detected.append(label)

        cv2.rectangle(
            original,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            original,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagem Original")
        st.image(
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    with col2:
        st.subheader("Imagem Segmentada")
        st.image(
            thresh,
            use_container_width=True,
            clamp=True
        )

    st.subheader("Resultado")

    if detected:

        for obj in detected:
            st.success(f"Objeto identificado: {obj}")

    else:

        st.warning("Nenhum objeto identificado.")

    st.info(f"Total de objetos detectados: {len(detected)}")