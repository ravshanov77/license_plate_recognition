import cv2 as cv

#drawing bbox
def draw_bbox(img, labels):

  x = 0
  while x < len(labels):
    for i in labels:
      img = cv.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), color=[0,255,0], thickness=2)

      img = cv.putText(img, 'car', org=(int(i[0]), int(i[1]) - 5),
                       fontFace=int(img.shape[0] ** .15),
                       fontScale=int(img.shape[0] ** .1),
                       color=[255,0,0],
                       thickness=int(img.shape[0] ** .1))

      font = cv.FONT_HERSHEY_SIMPLEX
    x += 1

  return img