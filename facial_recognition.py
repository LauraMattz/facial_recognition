import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)
reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rosto = reconhecimento_rosto.FaceDetection()

while webcam.isOpened():
    validacao, frame = webcam.read()
    if not validacao:
        break
    imagem = frame
    lista_rostos = reconhecedor_rosto.process(imagem)
    
    h = 0  # Define um valor padrão para a variável h
    
    if lista_rostos.detections:
        # Conta o número de rostos detectados
        num_rostos = len(lista_rostos.detections)
        for rosto in lista_rostos.detections:
            caixa_delimitadora_relativa = rosto.location_data.relative_bounding_box
            x, y, w, h = int(caixa_delimitadora_relativa.xmin * imagem.shape[1]), int(caixa_delimitadora_relativa.ymin * imagem.shape[0] - w * 0.2), int(caixa_delimitadora_relativa.width * imagem.shape[1]), int(caixa_delimitadora_relativa.width * 1.5 * imagem.shape[0])
            
            # Desenha o retângulo verde com o título "Face"
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imagem, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Adiciona a contagem de rostos detectados à imagem
        cv2.putText(imagem, f"Rostos detectados: {num_rostos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    cv2.imshow("Rostos na sua webcam", imagem)
    if cv2.waitKey(5) == 27:
        break
        
webcam.release()
cv2.destroyAllWindows()
