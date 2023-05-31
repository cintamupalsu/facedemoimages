import cv2
import face_recognition as FR
font=cv2.FONT_HERSHEY_SIMPLEX

# カメラ設定
cam=cv2.VideoCapture(0)

okamuraKao=FR.load_image_file('/Users/s118301/Documents/python/demoimages/known/okamura_hisanori.jpg')
faceLoc=FR.face_locations(okamuraKao)[0]
okamuraKaoEncode=FR.face_encodings(okamuraKao)[0]

shigenagaKao=FR.load_image_file('/Users/s118301/Documents/python/demoimages/known/shigenaga_tomoko.jpg')
faceLoc=FR.face_locations(shigenagaKao)[0]
shigenagaKaoEncode=FR.face_encodings(shigenagaKao)[0]

knownEncodings=[okamuraKaoEncode,shigenagaKaoEncode]
names=['Okamura Hisanori','Shigenaga Tomoko']

# 無限ループで、カメラから写真を取得します。
while True:
    ignore, unknownKao = cam.read()
    
    # 画像の取得には OpenCV を使用するため、BGR 形式を使用する必要がありますが、顔認識ライブラリは RGB 形式で動作します。
    # BGR から RGB に反転する必要があります
    unknownKaoRGB=cv2.cvtColor(unknownKao,cv2.COLOR_BGR2RGB)
    faceLocations=FR.face_locations(unknownKaoRGB)
    unknownEncodings=FR.face_encodings(unknownKaoRGB,faceLocations)

    for faceLocation, unknownEncoding in zip(faceLocations, unknownEncodings):
        top,right,bottom,left=faceLocation
        cv2.rectangle(unknownKao,(left,top),(right,bottom),(255,0,0),2)
        name='Unknown Person'
        matches=FR.compare_faces(knownEncodings,unknownEncoding)
        print(matches)
        if True in matches:
            matchIndex=matches.index(True)
            name=names[matchIndex]
        cv2.putText(unknownKao,name,(left,top),font,.5,(0,0,255),2)

    cv2.imshow('my Window',unknownKao)
    # q キーを押してプログラムを終了するように設定します。
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
# カメラを放し、アクティブなウィンドウを閉じます。
cam.release()
cv2.destroyAllWindows