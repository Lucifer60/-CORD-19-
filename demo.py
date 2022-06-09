import tkinter as tk
from tkinter import ttk
import pandas as pd
from tkinter import messagebox
import time
import requests
from lxml import etree
from tkinter import filedialog
from demogetanswer import demogetdata
from PIL import Image,ImageTk
class CORDQA:
    def __init__(self, master):
        self.df_c = ['ANSWER', 'Score','TITLE','AUTHOR','PUBLISHTIME','QUESTION','Link']
        self.var1 = tk.StringVar()
        self.label1 = tk.Label(master, text='Please input your question:', font=('Consolas', 12))
        self.label1.grid()  # grid是网格布局
        self.entry1 = tk.Entry(master, textvariable=self.var1, font=('Consolas', 14), width=65)
        self.entry1.grid(row=0, column=1)

        self.button1 = tk.Button(master, text='Search', font=('Consolas', 12), command=self.frist)
        self.button1.grid(row=0, column=2)
        self.tree1 = ttk.Treeview(
            master,  #
            height=15,  # 表格显示的行数
            columns=self.df_c,  # 显示的列
            show='headings',  # 隐藏首列
        )

        for x in self.df_c:
            self.tree1.heading(x, text=x)
            self.tree1.column(x, width=120)
        self.tree1.grid(row=2, columnspan=3)  # columnspan=3合并单元格，横跨3列
        self.button4 = tk.Button(master, text='Save result', font=('Consolas', 12), command=self.save)
        self.button4.grid(row=3, column=1)

    def get_data(self):
        question = self.entry1.get()
        if question:
            data = demogetdata(question)
            # data = pd.read_csv(question+"/answer_result.csv")
            return data
        else:
            messagebox.showinfo(title='Error', message='Please input your answer')

    def show(self):
        for row in self.tree1.get_children():
            self.tree1.delete(row)
        self.df = self.get_data()
        # self.df_col = self.df.columns.tolist()
        # self.tree1['columns'] = self.df_col
        # for x in self.df_col:
        #     self.tree1.heading(x, text = x)
        #     self.tree1.column(x, width=100)
        # 在treeview中显示dataframe数据
        for i in range(len(self.df)):
            self.tree1.insert('', i, values=self.df.iloc[i, :].tolist())

    def frist(self):
        self.show()


    def save(self):
        try:
            savefile = filedialog.asksaveasfilename(filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
            self.df.to_excel(savefile + ".xlsx", index=False)

        except Exception as e:
            messagebox.showerror(title='错误信息', message=str(e))


root = tk.Tk()
root.title('北京理工大学文本挖掘_邱小尧-CORD-19问答系统')
width,height = 998,432

root.geometry(str(width)+'x'+str(height)+"+200+100")
root.resizable(False, False)  # 设置窗口不可变
CORDQA(root)
root.mainloop()