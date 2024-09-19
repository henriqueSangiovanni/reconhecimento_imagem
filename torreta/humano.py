import cv2

# Carrega os arquivos do modelo pré-treinado (MobileNet SSD)
prototxt_path = 'deploy.prototxt'
model_path = 'mobilenet_iter_73000.caffemodel'

# Inicializa a rede neural com os arquivos do modelo
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Captura frame por frame
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível acessar a webcam")
        break

    # Obtém as dimensões do frame
    (h, w) = frame.shape[:2]

    # Redimensiona o frame e cria um blob para a rede neural
    #o que é um blob https://learnopencv.com/blob-detection-using-opencv-python-c/
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Realiza a detecção
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Obtém o índice da classe detectada (15 é humano)
            idx = int(detections[0, 0, i, 1])
            if idx == 15:
                # Obtém as coordenadas da caixa delimitadora
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Desenha um retângulo ao redor da silhueta humana detectada
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f'Pessoa: {confidence:.2f}', (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                centerX = (startX + endX) // 2
                centerY = (startY + endY) // 2
                cv2.circle(frame, (centerX, centerY), 10, (0, 0, 255), -1)
                print(f'Centro do círculo: ({centerX}, {centerY})')


    cv2.imshow('Detecção de Silhueta Humana', frame)

    # Pressione 'e' para sair
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()