# 指定包含要合并的文件的目录
directory_path = 'D:/PyCharm 2023.2.1/learn/bin2txt/AE2'
# 打开一个文本文件用于写入
output_file_name='label_20_40.txt'
label_path=f'{directory_path}/{output_file_name}'
with open(label_path, 'w') as file:
    # 对于每个数字从0到19
    for number in range(20):
        # 写入这个数字200次，每次写入后换行
        for _ in range(40):
            file.write(str(number) + '\n')

print("文件写入完成。")
