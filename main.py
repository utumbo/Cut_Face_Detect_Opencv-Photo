import cv2

#загрузка классификатора
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#загрузка изображения
img = cv2.imread('face.jpg')

#преобразование в серый
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#обнаружение лиц
face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#прямоугольники вокруг лиц
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

#отображение результата
cv2.imshow('Detected_face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()