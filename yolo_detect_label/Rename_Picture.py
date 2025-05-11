"""
此文件的作用是给文件夹里面的所有文件进行数字命名排序从0到n
用法:
填写path变量然后运行脚本就可以
"""
import os
#填写图像路径
path = "C:\\Users\\sunxi\\Documents\\地下城与勇士\\ScreenShot"

file_names = os.listdir(path)
print(file_names)

for i , file_names in enumerate(file_names,):
    # structure full of images path
    # 构造       完整  的  图像   路径
    old_file_path = os.path.join(path, file_names)
    new_file_name = f"{i+1273}.jpg"
    # sava to path inside,new_file_name is neo file name
    # 保存到path    里                    是 新的 文件名
    new_file_path = os.path.join(path, new_file_name)
    # rechristen file
    os.rename(old_file_path, new_file_path)
print("处理完成")