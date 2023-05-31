import cv2
import face_recognition as FR
font=cv2.FONT_HERSHEY_SIMPLEX
okamuraKao=FR.load_image_file('/Users/s118301/Documents/python/demoimages/known/okamura_hisanori.jpg')
faceLoc=FR.face_locations(okamuraKao)[0]
okamuraKaoEncode=FR.face_encodings(okamuraKao)[0]
shigenagaKao=FR.load_image_file('/Users/s118301/Documents/python/demoimages/known/shigenaga_tomoko.jpg')
faceLoc=FR.face_locations(shigenagaKao)[0]
shigenagaKaoEncode=FR.face_encodings(shigenagaKao)[0]
knownEncodings=[okamuraKaoEncode,shigenagaKaoEncode]
names=['Okamura Hisanori','Shigenaga Tomoko']
unknownKao=FR.load_image_file('/Users/s118301/Documents/python/demoimages/unknown/u1.jpg')
unknownKaoBGR=cv2.cvtColor(unknownKao,cv2.COLOR_RGB2BGR)
faceLocations=FR.face_locations(unknownKao)
unknownEncodings=FR.face_encodings(unknownKao,faceLocations)
for faceLocation, unknownEncoding in zip(faceLocations, unknownEncodings):
    top,right,bottom,left=faceLocation
    cv2.rectangle(unknownKaoBGR,(left,top),(right,bottom),(255,0,0),2)
    name='Unknown Person'
    matches=FR.compare_faces(knownEncodings,unknownEncoding)
    print(matches)
    if True in matches:
        matchIndex=matches.index(True)
        name=names[matchIndex]
    cv2.putText(unknownKaoBGR,name,(left,top),font,.5,(0,0,255),2)
cv2.imshow('my Window',unknownKaoBGR)
cv2.waitKey(5000)