def findNumber(arr, k):
    if k in arr:
        return "YES"
    else:
        return "NO"


if __name__ == '__main__':
    data = list(map(int, input().rstrip().split()))
    n = data[0]           # length of array
    arr = data[1:n+1]     # actual array elements
    k = data[-1]          # element to search
    result = findNumber(arr, k)
    print(result)