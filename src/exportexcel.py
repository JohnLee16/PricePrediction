import pandas as pd

class WriteExcel():
    def __init__(self):
        pass

    def write_data_to_excel(data, xlsxpath):
        writer = pd.ExcelWriter(xlsxpath)
        data.to_excel(writer)

    def write_data_to_csv(data, csvpath):
        data.to_csv(csvpath)
