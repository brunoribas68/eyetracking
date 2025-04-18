from gaze_tracking import GazeTracking
import cv2

# Inicializa o rastreador de olhos
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    # Captura o quadro da webcam
    _, frame = webcam.read()

    # Analisa o quadro com a biblioteca GazeTracking
    gaze.refresh(frame)

    # Recupera o quadro processado para exibição
    frame = gaze.annotated_frame()

    # Determina a direção do olhar
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"
    else:
        text = "No gaze detected"

    # Exibe o texto na tela
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Marca os olhos com círculos
    left_pupil = gaze.pupil_left_coords()  # Coordenadas do olho esquerdo
    right_pupil = gaze.pupil_right_coords()  # Coordenadas do olho direito

    if left_pupil:
        cv2.circle(frame, left_pupil, 5, (255, 0, 0), -1)  # Desenha ponto no olho esquerdo
    if right_pupil:
        cv2.circle(frame, right_pupil, 5, (0, 0, 255), -1)  # Desenha ponto no olho direito

    # Exibe o quadro processado
    cv2.imshow("Eye Tracking", frame)

    # Fecha ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
webcam.release()
cv2.destroyAllWindows()

with open("eye_data.csv", "a") as file:
    file.write(f"{gaze.horizontal_ratio()}, {gaze.vertical_ratio()}\n")