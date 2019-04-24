
def shellquote(s):
    return "'" + s.replace("'", "'\\''") + "'"

aaa = shellquote(sign_names[y_train[sample_index]])
print(y_train[sample_index])
print(aaa)
mpimg.imsave(f'./data/image1.jpg', sample_image)






