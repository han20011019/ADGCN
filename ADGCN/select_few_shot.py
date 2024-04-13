import os

# 指定文件夹的路径
folder_path = 'D:/PyCharm 2023.2.1/learn/ADGCN/data preprocessing/3_ProcessedSession/TrimedSession/Train/BitTorrent-L7'
output_file = open('Few_Shot/BitTorrent.txt', 'w')

# 获取所有的.bin文件
all_bin_files = [f for f in os.listdir(folder_path) if f.endswith('.bin')]

# 初始化计数器
file_count = 0

# 遍历所有bin文件
for filename in all_bin_files:
    bin_file_path = os.path.join(folder_path, filename)

    # 打开bin文件并读取内容
    with open(bin_file_path, 'rb') as bin_file:
        file_content = bin_file.read()

    # 将每个字节转换为整数
    int_values = [int(byte) for byte in file_content]

    # 如果所有整数的值都为0，则放弃这个文件，继续下一个文件
    if all(value == 0 for value in int_values):
        print(f'文件 "{filename}" 的所有值均为0，放弃处理。')
        continue

    # 将每个整数写入文本文件，用空格隔开
    output_line = ' '.join(str(value) for value in int_values)
    output_file.write(output_line + '\n')  # 在每个bin文件读取完后添加换行符

    # 文件处理完毕，打印文件名
    print(f'文件 "{filename}" 已经完成。')

    # 计数器增加
    file_count += 1

    # 如果已经读取了200个文件，则退出循环
    if file_count >= 200:
        break

# 关闭文本文件
output_file.close()

# 打印处理文件的总数
print(f'总共处理了 {file_count} 个文件。')
