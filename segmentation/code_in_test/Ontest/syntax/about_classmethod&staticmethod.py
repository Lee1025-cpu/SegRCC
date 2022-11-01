# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2021-01-14 14:38
# 文件名称：about_classmethod&staticmethod
# 开发工具：PyCharm
class Student(object):

    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    @classmethod
    def from_string(cls, name_str):
        first_name, last_name = map(str, name_str.split(' '))
        student = cls(first_name, last_name)
        return student


class Dates(object):
    def __init__(self, date):
        self.date = date

    def getDate(self):
        return self.date

    @staticmethod
    def toDashDate(date):
        return date.replace("/", "-")


if __name__ == "__main__":
    scott = Student('Scott',  'Robinson')

    scott1 = Student.from_string('Scott Robinson')

    date = Dates("15-12-2016")
    dateFromDB = "15/12/2016"
    dateWithDash = Dates.toDashDate(dateFromDB)

    if (date.getDate() == dateWithDash):
        print("Equal")
    else:
        print("Unequal")