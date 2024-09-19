import cv2

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Captura o primeiro frame para referência
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

while True:
    # Captura o próximo frame
    ret, frame2 = cap.read()
    if not ret:
        print("Não foi possível acessar a webcam")
        break

    # Converte para escala de cinza e aplica desfoque
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

    # Calcula a diferença absoluta entre o primeiro frame e o atual
    diff_frame = cv2.absdiff(frame1_gray, frame2_gray)

    _, thresh_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)


    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignora pequenos movimentos
        if cv2.contourArea(contour) < 500:
            continue

        # Desenha um retângulo ao redor do movimento detectado
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Movimento', frame2)

    # Atualiza o frame
    frame1_gray = frame2_gray

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()