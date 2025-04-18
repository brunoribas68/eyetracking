import cv2
from gaze_tracking import GazeTracking

def main():
    # Inicializa o rastreador de olhos
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)  # Configura a webcam

    while True:
        # Captura o quadro da câmera
        _, frame = webcam.read()

        # Analisa o quadro
        gaze.refresh(frame)

        # Quadro anotado para exibição
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

        # Mostra o texto na tela
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibe a janela com o rastreamento
        cv2.imshow("Eye Tracking", frame)

        # Interrompe ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera recursos
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
