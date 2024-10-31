import cv2
import os

#загрузка каскада
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Папка для фото
image_folder = 'input_face'
output_folder = 'output_face'

#Создание папки для готового фото если её нет
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#получение  всех файлов в папке
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

#Обработка изображений
for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path)

    #Проверка загрузки изображения
    if img is None:
        print(f"failed to upload image: {img_path}")
        continue

    #Преображение в серый
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    #вырезаем и сохранем лица
    for i, (x, y, w, h) in enumerate(faces):
        face_img = img[y:y + h, x:x + w] #вырезаем лицо
        face_filename = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_face_{i + 1}.jpg")
        cv2.imwrite(face_filename, face_img)

    #отображение результата
    cv2.imshow('Detected_face', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()