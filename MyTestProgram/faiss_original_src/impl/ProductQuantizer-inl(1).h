/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace faiss {

inline PQEncoderGeneric::PQEncoderGeneric(
        uint8_t* code,
        int nbits,
        uint8_t offset) // offset 默认是 0，reg 无有效位，compute_codes_with_assign_index里可能不是0
        : code(code), offset(offset), nbits(nbits), reg(0) {
    assert(nbits <= 64);
    if (offset > 0) {
        reg = (*code & ((1 << offset) - 1)); // code 的低 offset 位放在 reg 的低 offset 位
    }
}

inline void PQEncoderGeneric::encode(uint64_t x) { // 保存 x 的低 nbits 位到 code 字节数组
    reg |= (uint8_t)(x << offset); // 取 x 的低 (8-offset) 位放在 reg 的高 (8-offset) 位
    x >>= (8 - offset); // x 的低 (8-offset) 位已经放到 reg 了，并且在下面可能放到 code 里
    if (offset + nbits >= 8) { // 大于等于 1 个字节了
        *code++ = reg; // 第一个字节

        // 第一个字节里存储了 x 的低 (8-offset) 位
        // 所以还剩下 x 的 (nbits - (8 - offset)) / 8 个整个的字节要存储
        // 最后 x 还有 (nbits - (8 - offset)) % 8 个有效位，保存到了 reg 里
        for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i) { 
            *code++ = (uint8_t)x;
            x >>= 8;
        }

        offset += nbits;
        offset &= 7; // 从已经存储了 offset 位的某个字节开始存储 x 的低 nbits 位，最后一个字节里会存储 (offset+nbits)%8 位
        reg = (uint8_t)x; // reg 存储 x 剩下的不足矣占满 1 个字节的位
    } else {
        offset += nbits; // 不到 1 个字节，此时 reg 的低 offset 位是有效位
    }
}

inline PQEncoderGeneric::~PQEncoderGeneric() {
    if (offset > 0) { // 懒存储，析构时才把最后多出的几个有效位放到 code 里
        *code = reg;
    }
}

inline PQEncoder8::PQEncoder8(uint8_t* code, int nbits) : code(code) {
    assert(8 == nbits);
}

inline void PQEncoder8::encode(uint64_t x) { // 只存储 8 位
    *code++ = (uint8_t)x; // code++有什么效果？decode也是++，没什么意义吧
}

inline PQEncoder16::PQEncoder16(uint8_t* code, int nbits)
        : code((uint16_t*)code) {
    assert(16 == nbits);
}

inline void PQEncoder16::encode(uint64_t x) {
    *code++ = (uint16_t)x;
}

inline PQDecoderGeneric::PQDecoderGeneric(const uint8_t* code, int nbits)
        : code(code),
          offset(0),
          nbits(nbits),
          mask((1ull << nbits) - 1),
          reg(0) {
    assert(nbits <= 64);
}

inline uint64_t PQDecoderGeneric::decode() {
    if (offset == 0) { // reg 里每一位都无效
        reg = *code;   // 保存当前 code 的 8 位
    }
    uint64_t c = (reg >> offset); // reg 的低 offset 位是无效位，此时 c 的高 (8-offset) 位是有效位

    if (offset + nbits >= 8) { // 需要取的位跨越了多个字节
        uint64_t e = 8 - offset; // code 当前字节里的有效位数（已经取出了 e 个位到 c 里）
        ++code;
        for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i) { // 还需要取多少个整个的字节
            c |= ((uint64_t)(*code++) << e); // 把 code 当前的 8 位放到 c 的低 e+1 到 e+8 位上
            e += 8; // 又保存了 8 个位到 c 里
        }

        offset += nbits;
        offset &= 7;
        if (offset > 0) { // 是否还有剩余的位没保存到 c 里
            reg = *code;
            c |= ((uint64_t)reg << e);
        }
    } else {
        offset += nbits;
    }

    return c & mask; // 只保留低 nbits 位，高于 nbits 的位置零
}

inline PQDecoder8::PQDecoder8(const uint8_t* code, int nbits_in) : code(code) {
    assert(8 == nbits_in);
}

inline uint64_t PQDecoder8::decode() {
    return (uint64_t)(*code++);
}

inline PQDecoder16::PQDecoder16(const uint8_t* code, int nbits_in)
        : code((uint16_t*)code) {
    assert(16 == nbits_in);
}

inline uint64_t PQDecoder16::decode() {
    return (uint64_t)(*code++);
}

} // namespace faiss
