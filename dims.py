from random import randint


# Hi, Wi, Ki, Ko, ks, g, p, s

dimslist_direct = [
    # 3x3
    [ 3,  3,    1,  32, 3, 1, 0, 1],
    [ 3,  3,    8,  32, 3, 1, 0, 1],
    [ 3,  3,   16,  32, 3, 1, 0, 1],
    [ 3,  3,   32,  32, 3, 1, 0, 1],
    [ 3,  3,   48,  32, 3, 1, 0, 1],
    [ 3,  3,   16,  64, 3, 1, 0, 1],
    [ 6,  6,   16,  32, 3, 1, 0, 1],
    [ 6, 36,   32,  32, 3, 1, 0, 1],
    [15, 36,   32,  32, 3, 1, 0, 1],
    [15, 15,  128,  64, 3, 1, 0, 1],
    [15, 16,  128,  64, 3, 1, 0, 1],
    [17, 17,  128,  64, 3, 1, 0, 1],
    [17, 17,  128,  61, 3, 1, 0, 1],
    [17, 17,  127,  61, 3, 1, 0, 1],
    [ 3,  6, 1024,  32, 3, 1, 0, 1],
    [ 3,  6,  256, 256, 3, 1, 0, 1],
    # 1x1
    [ 3,  3,    1,  32, 1, 1, 0, 1],
    [ 3,  3,    8,  32, 1, 1, 0, 1],
    [ 3,  3,   16,  32, 1, 1, 0, 1],
    [ 3,  3,   32,  32, 1, 1, 0, 1],
    [ 3,  3,   48,  32, 1, 1, 0, 1],
    [ 3,  3,   16,  64, 1, 1, 0, 1],
    [ 6,  6,   16,  32, 1, 1, 0, 1],
    [ 6, 36,   32,  32, 1, 1, 0, 1],
    [15, 36,   32,  32, 1, 1, 0, 1],
    [15, 15,  128,  64, 1, 1, 0, 1],
    [15, 16,  128,  64, 1, 1, 0, 1],
    [17, 17,  128,  64, 1, 1, 0, 1],
    [17, 17,  128,  61, 1, 1, 0, 1],
    [17, 17,  127,  61, 1, 1, 0, 1],
    [ 3,  6, 1024,  32, 1, 1, 0, 1],
    [ 3,  6,  256, 256, 1, 1, 0, 1]
]


dimslist_direct_dw = [
    [ 3,  3,    1,    1, 3,    1, 0, 1],
    [ 3,  3,    8,    8, 3,    8, 0, 1],
    [ 3,  3,   16,   16, 3,   16, 0, 1],
    [ 3,  3,   32,   32, 3,   32, 0, 1],
    [ 3,  3,   48,   48, 3,   48, 0, 1],
    [ 3,  3,   64,   64, 3,   64, 0, 1],
    [ 6,  6,   16,   16, 3,   16, 0, 1],
    [ 6, 36,   32,   32, 3,   32, 0, 1],
    [15, 36,   32,   32, 3,   32, 0, 1],
    [15, 15,  128,  128, 3,  128, 0, 1],
    [17, 17,  128,  128, 3,  128, 0, 1],
    [17, 17,   61,   61, 3,   61, 0, 1],
    [17, 17,  127,  127, 3,  127, 0, 1],
    [ 3,  6,  256,  256, 3,  256, 0, 1],
    [ 3,  3, 1024, 1024, 3, 1024, 0, 1],
    [ 3,  6, 1024, 1024, 3, 1024, 0, 1]
]


dimslist_direct_l3_l2_tiled = [
    [14, 14, 256, 256, 3, 1, 0, 1],
    [56, 56,  96,  96, 3, 1, 0, 1],
    [10, 10, 512, 512, 1, 1, 0, 1],
    [30, 80, 128, 128, 1, 1, 0, 1]
]


def dimslist_random(n, ks, dw=False):
    ch = randint(1, 200) if dw else None
    g = ch if dw else 1

    def dims_random(ch, max_ch=200, max_sp=10):
        if ch is None:
            return [randint(1, max_ch), randint(1, max_ch), randint(3, max_sp), randint(3, max_sp)]
        else:
            return [ch, ch, randint(3, max_sp), randint(3, max_sp)]

    return [dims_random(ch) + [ks, g] for i in range(n)]


def MobileNetV2_bottleneck(hi, wi, ki, ko, t, s):
    return [
        #Hi     , Wi     ,     Ki,     Ko, ks,      g, p, s
        [hi     , wi     ,     ki, t * ki,  1,      1, 0, 1],
        [hi     , wi     , t * ki, t * ki,  3, t * ki, 1, s],
        [hi // s, wi // s, t * ki,     ko,  1,      1, 0, 1]
    ] + ([
        [hi     , wi     , t * ki, t * ki,  3, t * ki, 1, 1],
        [hi     , wi     , t * ki,     ko,  1,      1, 0, 1]
    ] if s > 1 else [])


def MobileNetV2_dimslist():
    l = []
    l += [[224, 224, 3, 32, 3, 1, 1, 2]]
    l += MobileNetV2_bottleneck(112, 112,  32,  16, 1, 1)
    l += MobileNetV2_bottleneck(112, 112,  16,  24, 6, 2)
    l += MobileNetV2_bottleneck( 56,  56,  24,  32, 6, 2)
    l += MobileNetV2_bottleneck( 28,  28,  32,  64, 6, 2)
    l += MobileNetV2_bottleneck( 14,  14,  64,  96, 6, 1)
    l += MobileNetV2_bottleneck( 14,  14,  96, 160, 6, 2)
    l += MobileNetV2_bottleneck(  7,   7, 160, 320, 6, 1)
    l += [[7, 7, 320, 1280, 1, 1, 0, 1]]
    return l


def MobileNetV1_block(input_size, ki, ko, s):
    return [
        [input_size     , input_size     , ki, ki, 3, ki, 1, s],
        [input_size // s, input_size // s, ki, ko, 1,  1, 0, 1]
    ]


def MobileNetV1_dimslist(input_size):
    l = []
    l += [[input_size, input_size, 3, 32, 3, 1, 1, 2]]
    l += MobileNetV1_block(input_size //  2,   32,   64, 1)
    k = 64
    h = input_size // 2
    for i in range(4):
        l += MobileNetV1_block(h, k, 2*k, 2); h //= 2; k *= 2
        l += MobileNetV1_block(h, k,   k, 1)
    return l


def ResNetTinyML_dimslist():
    l = []
    l += [[32, 32,  3, 16, 3, 1, 1, 1]]
    l += [[32, 32, 16, 16, 3, 1, 1, 1]]
    l += [[32, 32, 16, 32, 3, 1, 1, 2]]
    l += [[16, 16, 32, 32, 3, 1, 1, 1]]
    l += [[16, 16, 32, 64, 3, 1, 1, 2]]
    l += [[ 8,  8, 64, 64, 3, 1, 1, 1]]
    return l


def ResNet_block(output_size, ko):
    return [
        [2 * output_size, 2 * output_size, ko // 2, ko, 3, 1, 1, 2],
        [    output_size,     output_size,      ko, ko, 3, 1, 1, 1]
    ]


def ResNet18_dimslist():
    l = []
    # l+= [[224, 224, 3, 64, 7, 1, 1, 2]]
    # l+= maxpool(kernel_shape=(3, 3), stride=2)
    l += [[56, 56, 64, 64, 3, 1, 1, 1]]
    output_size = 28
    ko = 128
    for i in range(3):
        l += ResNet_block(output_size, ko)
        output_size //= 2; ko *= 2

    return l


perfect_3x3_nol3_dims = [
    [26, 26, 256,  32, 3, 1, 0, 1],
    [29, 29, 256,  32, 3, 1, 0, 1],
    [26, 26, 128, 128, 3, 1, 0, 1],
    [29, 29, 128, 128, 3, 1, 0, 1],
    [11, 74, 128, 128, 3, 1, 0, 1],
    [5, 167, 128, 128, 3, 1, 0, 1],
    [74, 11, 128, 128, 3, 1, 0, 1],
    [167, 5, 128, 128, 3, 1, 0, 1]
]


perfect_3x3_nol3_large_ki_dims = [
    [5, 5,  64, 32, 3, 1, 0, 1],
    [5, 5, 128, 32, 3, 1, 0, 1],
    [5, 5, 144, 32, 3, 1, 0, 1],
    [5, 5, 176, 32, 3, 1, 0, 1],
    [5, 5, 224, 32, 3, 1, 0, 1],
    [5, 5, 240, 32, 3, 1, 0, 1],
    [5, 5, 256, 32, 3, 1, 0, 1]
]


imperfect_3x3_nol3_dims = [
    [ 24,  24, 241,  33, 3, 1, 0, 1],
    [ 27,  27, 241,  33, 3, 1, 0, 1],
    [ 24,  24, 113, 129, 3, 1, 0, 1],
    [ 27,  27, 113, 129, 3, 1, 0, 1],
    [ 12,  75, 113,  97, 3, 1, 0, 1],
    [  6, 168, 113,  97, 3, 1, 0, 1],
    [ 75,  12, 113,  97, 3, 1, 0, 1],
    [168,   6, 113,  97, 3, 1, 0, 1]
]


perfect_1x1_nol3_dims = [
    [30, 30, 256,  32, 1, 1, 0, 1],
    [33, 33, 256,  32, 1, 1, 0, 1],
    [30, 30, 128, 128, 1, 1, 0, 1],
    [33, 33, 128, 128, 1, 1, 0, 1],
    [12, 75, 128, 128, 1, 1, 0, 1],
    [6, 168, 128, 128, 1, 1, 0, 1],
    [75, 12, 128, 128, 1, 1, 0, 1],
    [168, 6, 128, 128, 1, 1, 0, 1]
]


perfect_1x1_nol3_large_ki_dims = [
    [3, 3,   64, 32, 1, 1, 0, 1],
    [3, 3,  128, 32, 1, 1, 0, 1],
    [3, 3,  256, 32, 1, 1, 0, 1],
    [3, 3,  512, 32, 1, 1, 0, 1],
    [3, 3, 1024, 32, 1, 1, 0, 1],
    [3, 3, 1280, 32, 1, 1, 0, 1],
    [3, 3, 1536, 32, 1, 1, 0, 1],
    [3, 3, 2048, 32, 1, 1, 0, 1]
]

imperfect_1x1_nol3_dims = [
    [31, 31, 257,  33, 1, 1, 0, 1],
    [34, 34, 257,  33, 1, 1, 0, 1],
    [31, 31, 129, 129, 1, 1, 0, 1],
    [34, 34, 129, 129, 1, 1, 0, 1],
    [13, 76, 129, 129, 1, 1, 0, 1],
    [7, 169, 129, 129, 1, 1, 0, 1],
    [76, 13, 129, 129, 1, 1, 0, 1],
    [169, 7, 129, 129, 1, 1, 0, 1]
]

components_tests = [
    ## 3x3 tests

    ### variable Hi
    [5, 5, 32, 32, 3, 1, 0, 1],
    [20, 5, 32, 32, 3, 1, 0, 1],
    [50, 5, 32, 32, 3, 1, 0, 1],
    [100, 5, 32, 32, 3, 1, 0, 1],
    [200, 5, 32, 32, 3, 1, 0, 1],
    [400, 5, 32, 32, 3, 1, 0, 1],

    ### variable Wi
    [5, 20, 32, 32, 3, 1, 0, 1],
    [5, 50, 32, 32, 3, 1, 0, 1],
    [5, 100, 32, 32, 3, 1, 0, 1],
    [5, 200, 32, 32, 3, 1, 0, 1],
    [5, 400, 32, 32, 3, 1, 0, 1],

    ### variable Ki
    [5, 5, 64, 32, 3, 1, 0, 1],
    [5, 5, 80, 32, 3, 1, 0, 1],
    [5, 5, 128, 32, 3, 1, 0, 1],
    [5, 5, 200, 32, 3, 1, 0, 1],
    [5, 5, 256, 32, 3, 1, 0, 1],
    [5, 5, 350, 32, 3, 1, 0, 1],
    [5, 5, 512, 32, 3, 1, 0, 1],

    ### variable Ko
    [5, 5, 32,  64, 3, 1, 0, 1],
    [5, 5, 32,  80, 3, 1, 0, 1],
    [5, 5, 32, 128, 3, 1, 0, 1],
    [5, 5, 32, 200, 3, 1, 0, 1],
    [5, 5, 32, 256, 3, 1, 0, 1],
    [5, 5, 32, 350, 3, 1, 0, 1],
    [5, 5, 32, 512, 3, 1, 0, 1],

    ### all
    [ 5,  5,  32,  32, 3, 1, 0, 1],
    [10, 10,  64,  64, 3, 1, 0, 1],
    [15, 15,  80,  80, 3, 1, 0, 1],
    [20, 20, 128, 128, 3, 1, 0, 1],
    [30, 30, 200, 200, 3, 1, 0, 1],
    [60, 60, 256, 256, 3, 1, 0, 1],

    ### spatial up, channels down
    [  7,   7, 512, 512, 3, 1, 0, 1],
    [ 14,  14, 256, 256, 3, 1, 0, 1],
    [ 28,  28, 128, 128, 3, 1, 0, 1],
    [ 56,  56,  64,  64, 3, 1, 0, 1],
    [112, 112,  32,  32, 3, 1, 0, 1],
    [224, 224,  16,  16, 3, 1, 0, 1],
    [224, 224,   3,   3, 3, 1, 0, 1],


    ## 1x1 tests

    ### variable Hi
    [  5, 5, 32, 32, 1, 1, 0, 1],
    [ 20, 5, 32, 32, 1, 1, 0, 1],
    [ 50, 5, 32, 32, 1, 1, 0, 1],
    [100, 5, 32, 32, 1, 1, 0, 1],
    [200, 5, 32, 32, 1, 1, 0, 1],
    [400, 5, 32, 32, 1, 1, 0, 1],

    ### variable Wi
    [5,  20, 32, 32, 1, 1, 0, 1],
    [5,  50, 32, 32, 1, 1, 0, 1],
    [5, 100, 32, 32, 1, 1, 0, 1],
    [5, 200, 32, 32, 1, 1, 0, 1],
    [5, 400, 32, 32, 1, 1, 0, 1],

    ### variable Ki
    [5, 5,  64, 32, 1, 1, 0, 1],
    [5, 5,  80, 32, 1, 1, 0, 1],
    [5, 5, 128, 32, 1, 1, 0, 1],
    [5, 5, 200, 32, 1, 1, 0, 1],
    [5, 5, 256, 32, 1, 1, 0, 1],
    [5, 5, 350, 32, 1, 1, 0, 1],
    [5, 5, 512, 32, 1, 1, 0, 1],
    [5, 5, 1024, 32, 1, 1, 0, 1],

    ### variable Ko
    [5, 5, 32,  64, 1, 1, 0, 1],
    [5, 5, 32,  80, 1, 1, 0, 1],
    [5, 5, 32, 128, 1, 1, 0, 1],
    [5, 5, 32, 200, 1, 1, 0, 1],
    [5, 5, 32, 256, 1, 1, 0, 1],
    [5, 5, 32, 350, 1, 1, 0, 1],
    [5, 5, 32, 512, 1, 1, 0, 1],
    [5, 5, 32, 1024, 1, 1, 0, 1],

    ### all
    [ 5,  5,  32,  32, 1, 1, 0, 1],
    [10, 10,  64,  64, 1, 1, 0, 1],
    [15, 15,  80,  80, 1, 1, 0, 1],
    [20, 20, 128, 128, 1, 1, 0, 1],
    [30, 30, 200, 200, 1, 1, 0, 1],
    [60, 60, 256, 256, 1, 1, 0, 1],

    ### spatial up, channels down
    [  7,   7, 2048, 2048, 1, 1, 0, 1],
    [  7,   7, 1024, 1024, 1, 1, 0, 1],
    [  7,   7,  512,  512, 1, 1, 0, 1],
    [ 14,  14,  256,  256, 1, 1, 0, 1],
    [ 28,  28,  128,  128, 1, 1, 0, 1],
    [ 56,  56,   64,   64, 1, 1, 0, 1],
    [112, 112,   32,   32, 1, 1, 0, 1],
    [224, 224,   16,   16, 1, 1, 0, 1],
    [224, 224,    3,    3, 1, 1, 0, 1],


    ## dw tests

    ### variable Hi
    [  5, 5, 32, 32, 3, 32, 0, 1],
    [ 20, 5, 32, 32, 3, 32, 0, 1],
    [ 50, 5, 32, 32, 3, 32, 0, 1],
    [100, 5, 32, 32, 3, 32, 0, 1],
    [200, 5, 32, 32, 3, 32, 0, 1],
    [400, 5, 32, 32, 3, 32, 0, 1],

    ### variable Wi
    [5,  20, 32, 32, 3, 32, 0, 1],
    [5,  50, 32, 32, 3, 32, 0, 1],
    [5, 100, 32, 32, 3, 32, 0, 1],
    [5, 200, 32, 32, 3, 32, 0, 1],
    [5, 400, 32, 32, 3, 32, 0, 1],

    ### variable Ki
    [5, 5,  64,  64, 3,  64, 0, 1],
    [5, 5,  80,  80, 3,  80, 0, 1],
    [5, 5, 128, 128, 3, 128, 0, 1],
    [5, 5, 200, 200, 3, 200, 0, 1],
    [5, 5, 256, 256, 3, 256, 0, 1],
    [5, 5, 350, 350, 3, 350, 0, 1],
    [5, 5, 512, 512, 3, 512, 0, 1],

    ### all
    [ 5,  5,  32,  32, 3,  32, 0, 1],
    [10, 10,  64,  64, 3,  64, 0, 1],
    [15, 15,  80,  80, 3,  80, 0, 1],
    [20, 20, 128, 128, 3, 128, 0, 1],
    [30, 30, 200, 200, 3, 200, 0, 1],
    [60, 60, 256, 256, 3, 256, 0, 1],

    ### spatial up, channels down
    [  7,   7, 512, 512, 3, 512, 0, 1],
    [ 14,  14, 256, 256, 3, 256, 0, 1],
    [ 28,  28, 128, 128, 3, 128, 0, 1],
    [ 56,  56,  64,  64, 3,  64, 0, 1],
    [112, 112,  32,  32, 3,  32, 0, 1],
    [224, 224,  16,  16, 3,  16, 0, 1],
    [224, 224,   3,   3, 3,   3, 0, 1],
]