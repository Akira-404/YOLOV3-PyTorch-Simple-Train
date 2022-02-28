class Test():
    def __init__(self):
        self.a = 0

    def prepare(self):
        self.a = 1

    def get_a(self):
        return self.a


if __name__ == '__main__':
    t = Test()
    print(t.a)
    t.prepare()
    print(t.a)
