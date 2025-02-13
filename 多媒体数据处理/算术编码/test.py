from decimal import Decimal
from copy import deepcopy


# 获取用户输入的字符
def getChar():
    while (1):
        print('请输入字符，各字符间以空格分隔：')
        char = input().split(' ')
        # 只允许输入单字符，否则重新输入
        if (len(char[0]) == 1):
            break
    return char


# 获取用户输入的字符概率
def getProbability(char):
    while (1):
        print('请输入各字符的概率：')
        probability = input().split(' ')
        # 字符转十进制精确浮点
        probability = list(map(Decimal, probability))
        # 检查概率和是否为1并且概率列表元素个数是否与char列表的元素个数相等，否则重新输入
        if ((len(probability) == len(char)) & (sum(probability) == 1.0)):
            break
    return probability


# 获取用户输入的待编码字符串
def getString(char):
    while (1):
        print('请输入需要进行算数编码的字符串：')
        str = input()
        # 检查输入字符串中的字符是否均是字典中的字符，若否，则重新输入
        # all()检查iterable对象中的每个元素是否都满足bool判别式
        if (all(chr in char for chr in str)):
            break
    return str


# 构造概率字典
def createDict(char, prob):
    # 初始化概率字典
    probDict = {}
    # 记总字符数为num
    num = len(char)
    # 构造字符概率字典
    for i in range(num):
        if (i == 0):
            probDict[char[i]] = [Decimal('0.0'), prob[i]]  # 第一个字符的概率字典下界起点为0.0，上界为0.0+该路
        else:
            probDict[char[i]] = [probDict[char[i - 1]][1], probDict[char[i - 1]][1] + prob[i]]
            # 第二个数的下界是上一个字符的上界，上界是自己的下界+概率
    return probDict


# 算术编码函数，打印出最终区间的上下界
def arithEncode(string, probDict):
    lower_bound = Decimal('0.0')
    upper_bound = Decimal('1.0')
    for chr in string:
        intervalLength = upper_bound - lower_bound  # 区间长度
        # 不断更新区间上下界，注意必须先更新上界，否则会导致上界更新错误（因为上界的计算用的是上一次的下界）
        upper_bound = lower_bound + intervalLength * probDict[chr][1]
        lower_bound = lower_bound + intervalLength * probDict[chr][0]
        print(lower_bound, upper_bound)
    # 返回最终区间的上下界
    return lower_bound, upper_bound


# 求出最终区间内的最短二进制串
def dec2Bin(lower_bound, upper_bound):
    binStr = ''
    # 初始化01串转为的二进制数为0.0
    temp = Decimal(0.0)  # 储存当前2进制数代表的10进制数，解码的时候使用
    i = 1  # 初始化幂为1
    while (1):
        bit = 1
        # 若当前位置1得到的数大于区间上界，则将该位置0，temp不变
        if ((temp + Decimal(1 / (2 ** i) * bit)) > upper_bound):
            bit = 0
            binStr += '0'
        # 若当前位置1得到的数恰好在区间内（即恰好大于下界），则该位置1不变，然后直接出循环，无需生成后续的位数
        else:
            if ((temp + Decimal(1 / (2 ** i) * bit)) > lower_bound):
                binStr += '1'
                break
            # 若当前位置1得到的数小于下界，则该位置1
            else:
                binStr += '1'
        # 更新temp
        temp += Decimal(1 / (2 ** i) * bit)
        # 阶数自增1，下一轮循环时用
        i += 1
    return binStr


# 二进制串转十进制小数（乘二取整）
def bin2Dec(bin):
    dec = Decimal(0.0)
    for i in range(len(bin)):
        if (bin[i] == '1'):
            dec += Decimal(1 / (2 ** (i + 1)))
    # 这里返回的数据类型一定要是Decimal
    return dec


# 解码函数
def arithDecode(encodedBin, probDict, strLength):
    decodedStr = ''
    # probDict_copy存放原始的概率字典
    probDict_copy = deepcopy(probDict)
    # 二进制串转十进制小数
    encodedDec = bin2Dec(encodedBin)
    # 因为要解码出strLength个字符，所以共循环strLength次
    for _ in range(strLength):
        # 如果编码数字落在某个字符的概率区间内，则在结果中加入该字符
        for chr, interval in probDict.items():
            if ((encodedDec >= interval[0]) & (encodedDec < interval[1])):
                decodedStr += chr
                # 判断数字是否落在概率区间内，如果在，就继续对当前区间划分，看编码数字落在哪个字符对应的区间内
                # 更新字典中各字符的概率区间
                temp_lower_bound = interval[0]  # 记录当前区间下界和区间长度
                intervalLength = interval[1] - interval[0]
                for chr in probDict.keys():
                    probDict[chr][0] = temp_lower_bound + intervalLength * probDict_copy[chr][0]
                    probDict[chr][1] = temp_lower_bound + intervalLength * probDict_copy[chr][1]
                break
    return decodedStr


char = getChar()  # 输入字符
probability = getProbability(char)  # 输入字符对应的概率
string = getString(char)  # 输入待编码字符串
probDict = createDict(char, probability)  # 构造概率字典
print('概率字典如下：\n', probDict)

# 先求出最终区间的上下界
lower_bound, upper_bound = arithEncode(string, probDict)

# 再求出区间内的最短二进制串作为结果
binString = dec2Bin(lower_bound, upper_bound)
print('算术编码结果如下（最短二进制串）：\n', binString)
print('对应的十进制浮点数为：\n', bin2Dec(binString))

# 解码
decodedStr = arithDecode(binString, probDict, len(string))
print('算术解码结果如下：\n', decodedStr)
print('解码完成')
