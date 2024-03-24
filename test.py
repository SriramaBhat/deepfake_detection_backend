from PIL import Image
img = Image.open("./static/images/test1.png")
print(len(img.getpixel((0, 0))))
