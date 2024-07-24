import os
from datetime import datetime, timedelta

def readme_write():
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open("readme.md","a",encoding="utf-8") as f:
        print(current_time)
        answer = input("是否修改readme？")
        if answer == 'y':
            print("确定修改")
            f.write('\n修改时间：{time}    '.format(time = current_time))
        else:
            print('不修改readme内容')
            return
        while True:
            logs = input("向readme中加入：")
            if logs == '':
                print("Input finished.")
                break
            f.write(logs)
        

if __name__ == '__main__':
    readme_write()