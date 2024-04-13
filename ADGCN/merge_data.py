import numpy as np
# 指定包含要合并的文件的目录
directory_path = 'D:/PyCharm 2023.2.1/learn/ADGCN/process_by_AE'

# 指定要合并的文件列表
files_to_merge = ['BitTorrent_Dense_ae_40_784.txt', 'Facetime_Dense_ae_40_784.txt','FTP_Dense_ae_40_784.txt','Gmail_Dense_ae_40_784.txt','MySQL_Dense_ae_40_784.txt',
                  'Outlook_Dense_ae_40_784.txt','Skype_Dense_ae_40_784.txt','SMB-1_Dense_ae_40_784.txt','Weibo-1_Dense_ae_40_784.txt','WorldOfWarcraft_Dense_ae_40_784.txt'
                  ]
# 指定输出文件的名称
output_file_name = 'merged_data_Dense_ae_40_784_good.txt'
# 输出文件的完整路径
output_file_path = f'{directory_path}/{output_file_name}'
# 打开输出文件
with open(output_file_path, 'w') as output_file:
    # 遍历每个文件
    for file_name in files_to_merge:
        # 构建每个输入文件的完整路径
        input_file_path = f'{directory_path}/{file_name}'
        # 打开要合并的文件
        with open(input_file_path, 'r') as input_file:
            # 将文件内容写入输出文件
            output_file.write(input_file.read())
            # 在每个文件的内容后面加上一个换行符，以确保下一个文件内容从新的一行开始
            # output_file.write('\n')

print("文件合并完成。")
data = np.loadtxt('D:/PyCharm 2023.2.1/learn/ADGCN/process_by_AE/merged_data_Dense_ae_40_784_good.txt')
print(data.shape)