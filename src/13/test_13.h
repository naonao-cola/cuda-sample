#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
 * Initialize application state by reading inputs from the disk and
 * pre-allocating memory. Hand off to encrypt_decrypt to perform the actualy
 * encryption or decryption. Then, write the encrypted/decrypted results to
 * disk.
 * 通过从磁盘和读取输入来初始化应用程序状态预分配内存。移交给encrypt_decrypt来执行实际的
 * 加密或解密。然后，将加密/解密的结果写入磁盘。
 */
void test_encrypt();