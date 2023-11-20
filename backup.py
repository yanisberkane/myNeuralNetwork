#!/usr/bin/python3

import sys
import numpy as np
import math

EXIT_ERROR = 84
SUCCESS = 0
ALGOS = ["-xor", "-aes", "-rsa", "-pgp"]
MODE = ["-c", "-d"]
RSA_ONLY = ["-g"]
MODE_XOR_AES = ["-b"]
ALL_POSSIBLE_ARGS = ALGOS + MODE + RSA_ONLY + MODE_XOR_AES
LEN_AES_KEY = 16
args : dict
rcon = np.array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
                 0x40, 0x80, 0x1b, 0x36], dtype=np.uint8)

sbox = np.array([
                    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
                    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
                    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
                    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
                    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
                    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
                    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
                    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
                    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
                    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
                    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
                    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
                    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
                    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
                    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
                    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
                ], dtype=np.uint8)

inverse_sbox = np.array([
                    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
                    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
                    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
                    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
                    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
                    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
                    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
                    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
                    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
                    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
                    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
                    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
                    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
                    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
                    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
                    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
                ], dtype=np.uint8)

def multiply_by_2(byte : int):
    res = (byte << 1) & 0xff
    if (byte & 0x80):
        return res ^ 0x1b
    return res

def multiply_by_3(byte : int):
    return multiply_by_2(byte) ^ byte

def mix_column(column):
    return np.array([
        multiply_by_2(column[0]) ^ multiply_by_3(column[1]) ^ column[2] ^ column[3],
        column[0] ^ multiply_by_2(column[1]) ^ multiply_by_3(column[2]) ^ column[3],
        column[0] ^ column[1] ^ multiply_by_2(column[2]) ^ multiply_by_3(column[3]),
        multiply_by_3(column[0]) ^ column[1] ^ column[2] ^ multiply_by_2(column[3])
    ])

def mix_columns(matrix) -> np.ndarray:
    for i in range(4):
        matrix[i] = mix_column(matrix[i])
    return matrix

def run_xor():
    global args
    input_message = ""

    try:
        input_message = input()
    except EOFError:
        print("EOFError")
        sys.exit(EXIT_ERROR)

    try:
        message_bytes = bytes.fromhex(input_message)
        key_bytes = bytes.fromhex(args["KEY"])
    except ValueError:
        print("Invalid data")
        sys.exit(EXIT_ERROR)
    if (args["block"] == "-b"):
        if (len(message_bytes) < len(key_bytes)):
            print("Invalid data")
            sys.exit(EXIT_ERROR)
        if (len(message_bytes) > len(key_bytes)):
            message_bytes = message_bytes[:len(key_bytes)]
    repeated_key_bytes = (key_bytes * (len(message_bytes) // len(key_bytes) + 1))[:len(message_bytes)]
    result = bytes([a ^ b for a, b in zip(message_bytes, repeated_key_bytes)])
    print(result.hex())
    return

def get_input() -> bytes:
    input_message = ""

    try:
        input_message = input()
    except EOFError:
        print("EOFError")
        sys.exit(EXIT_ERROR)

    try:
        message_bytes = bytes.fromhex(input_message)
    except ValueError:
        print("Invalid data")
        sys.exit(EXIT_ERROR)
    return message_bytes

def expand_aes_matrix_key(previous_matrix_key : np.ndarray, nb_round : int) -> np.ndarray:
    new_matrix_key = np.zeros((4, 4), dtype=np.uint8)
    ## First column
    last_column = previous_matrix_key[3]
    last_column = np.roll(last_column, -1)
    last_column = np.array([sbox[b] for b in last_column])
    last_column[0] ^= rcon[nb_round]
    last_column ^= previous_matrix_key[0]
    new_matrix_key[0] = last_column
    ## Other columns
    for j in range(1, 4):
        new_matrix_key[j] = new_matrix_key[j - 1] ^ previous_matrix_key[j]
    return new_matrix_key

def encrypt_aes(encrypted_message_matrix : np.ndarray, round_key_matrix : np.ndarray, round : int) -> np.ndarray:
    encrypted_message_matrix = np.array([sbox[b] for b in encrypted_message_matrix.flatten()]).reshape((4, 4))
    encrypted_message_matrix = encrypted_message_matrix.transpose()
    for i in range(4):
        encrypted_message_matrix[i] = np.roll(encrypted_message_matrix[i], -i)
    encrypted_message_matrix = encrypted_message_matrix.transpose()
    if (round != 9):
        encrypted_message_matrix = mix_columns(encrypted_message_matrix)
    encrypted_message_matrix = encrypted_message_matrix ^ round_key_matrix
    return encrypted_message_matrix

def decrypt_aes(decrypt_message_matrix : np.ndarray, round_key_matrix : np.ndarray, round : int) -> np.ndarray:
    decrypt_message_matrix = decrypt_message_matrix.transpose()
    for i in range(4):
        decrypt_message_matrix[i] = np.roll(decrypt_message_matrix[i], i)
    decrypt_message_matrix = decrypt_message_matrix.transpose()
    decrypt_message_matrix = np.array([inverse_sbox[b] for b in decrypt_message_matrix.flatten()]).reshape((4, 4))
    decrypt_message_matrix = decrypt_message_matrix ^ round_key_matrix
    if (round != 0):
        for j in range(3):
            decrypt_message_matrix = mix_columns(decrypt_message_matrix)
    return decrypt_message_matrix

def advanced_encryption_standard(message_bytes : bytes, key_bytes : bytes) -> str:
    len_message_bytes = len(message_bytes)
    full_result = ""
    nb_tours = 1

    if (len(key_bytes) != LEN_AES_KEY):
        print("Invalid key")
        sys.exit(EXIT_ERROR)
    if (len(message_bytes) < LEN_AES_KEY):
        if (args["block"] == ""):
            message_bytes = message_bytes.ljust(LEN_AES_KEY, b'\x00')
        else:
            print("Invalid block message")
            sys.exit(EXIT_ERROR)
    if (len(message_bytes) > LEN_AES_KEY):
        full_message = message_bytes
        if (args["block"] == ""):
            nb_tours = len(full_message) // LEN_AES_KEY
        if (len(full_message) % LEN_AES_KEY != 0):
            full_message = full_message.ljust((len(full_message) // LEN_AES_KEY + 1) * LEN_AES_KEY, b'\x00')
    else:
        full_message = message_bytes

    for i in range (nb_tours):
        message_bytes = full_message[i * LEN_AES_KEY : (i + 1) * LEN_AES_KEY]

        # create a 2d columns matrix from the key
        key_matrix = np.frombuffer(key_bytes, dtype=np.uint8).reshape((4, 4))
        message_matrix = np.frombuffer(message_bytes, dtype=np.uint8).reshape((4, 4))

        # add round key
        if (args["mode"] == "-c"):
            encrypted_message_matrix = message_matrix ^ key_matrix

        previous_matrix = key_matrix
        all_keys_matrix = np.array([key_matrix])
        for i in range(10):
            new_key_matrix = expand_aes_matrix_key(previous_matrix, i)
            # MESSAGE ENCRYPTION:
            if (args["mode"] == "-c"):
                encrypted_message_matrix = encrypt_aes(encrypted_message_matrix, new_key_matrix, i)
                previous_matrix = new_key_matrix
            if (args["mode"] == "-d"):
                all_keys_matrix = np.append(all_keys_matrix, [new_key_matrix], axis=0)
                previous_matrix = new_key_matrix

        if (args["mode"] == "-c"):
            full_result += encrypted_message_matrix.flatten().tobytes().hex()

        # MESSAGE DECRYPTION:
        if (args["mode"] == "-d"):
            decrypt_message_matrix = message_matrix ^ all_keys_matrix[10]
            for i in range(9, -1, -1):
                decrypt_message_matrix = decrypt_aes(decrypt_message_matrix, all_keys_matrix[i], i)
            full_result += decrypt_message_matrix.flatten().tobytes().hex()
    return full_result

def run_aes():
    try:
        key_bytes = bytes.fromhex(args["KEY"])
    except ValueError:
        print("Invalid key")
        sys.exit(EXIT_ERROR)
    message_bytes = get_input()
    full_result = advanced_encryption_standard(message_bytes, key_bytes)
    print(full_result)
    return

def check_prime(n : int) -> bool:
    if n <= 1:
        return False
    if n % 2 == 0:
        return False
    return True

def generate_rsa_key(p : int, q : int):
    n = p * q
    totient = math.lcm(p - 1, q - 1)
    if (totient > 65537):
        public_key = 65537
    else:
        public_key = 257

    private_key = pow(public_key, -1, totient)
    return (public_key, private_key, n)

def big_endian_to_little_endian(n : int) -> int:
    return int.from_bytes(n.to_bytes((n.bit_length() + 7) // 8, 'big'), 'little')

def little_endian_to_big_endian(n : int) -> int:
    return int.from_bytes(n.to_bytes((n.bit_length() + 7) // 8, 'little'), 'big')

def run_rsa():
    if (args["rsa"] == "-g"):
        try:
            p = bytes.fromhex(args["P"])
            q = bytes.fromhex(args["Q"])
        except ValueError:
            print("Invalid p or q")
            sys.exit(EXIT_ERROR)
        try:
            p_value = little_endian_to_big_endian(int(p.hex(), 16))
            q_value = little_endian_to_big_endian(int(q.hex(), 16))
        except ValueError:
            print("Invalid data")
            sys.exit(EXIT_ERROR)
        # chack if p and q are prime
        if (not check_prime(p_value) or not check_prime(q_value)):
            print("p and q must be prime")
            sys.exit(EXIT_ERROR)
        # generate public and private key
        public_key, private_key, n = generate_rsa_key(p_value, q_value)
        print("public key: 0" + str(hex(big_endian_to_little_endian(public_key)))[2:] + "-" + hex(big_endian_to_little_endian(n))[2:])
        print("private key: " + str(hex(big_endian_to_little_endian(private_key))[2:]) + "-" + hex(big_endian_to_little_endian(n))[2:])
        return
    if (args["mode"] == "-d" or args["mode"] == "-c"):
        message = get_input()
        try:
            message_int = little_endian_to_big_endian(int(message.hex(), 16))
        except ValueError:
            print("Invalid data")
            sys.exit(EXIT_ERROR)
        if (args["mode"] == "-c" or args["mode"] == "-d"):
            try:
                key = args["KEY"].split("-")[0]
                n = args["KEY"].split("-")[1]
            except IndexError:
                print("Invalid data")
                sys.exit(EXIT_ERROR)
            try:
                key_int = little_endian_to_big_endian(int(key, 16))
                n_int = little_endian_to_big_endian(int(n, 16))
            except ValueError:
                print("Invalid data")
                sys.exit(EXIT_ERROR)
            result = pow(message_int, key_int, n_int)
            print(hex(big_endian_to_little_endian(result))[2:])
    return

def run_pgp():
    try:
        key = args["KEY"].split("-")[0]
        n = args["KEY"].split("-")[1]
    except IndexError:
        print("Invalid data")
        sys.exit(EXIT_ERROR)
    try:
        key_int = little_endian_to_big_endian(int(key, 16))
        n_int = little_endian_to_big_endian(int(n, 16))
    except ValueError:
        print("Invalid data")
        sys.exit(EXIT_ERROR)
    if (args["mode"] == "-c"):
        message = get_input()
        gen_128_bits = np.random.randint(0, 256, size=(16), dtype=np.uint8)
        # convert to int
        gen_128_bits_int = int(gen_128_bits.tobytes().hex(), 16)
        encrypted_key = pow(gen_128_bits_int, key_int, n_int)
        # print gen_128_bits
        print(hex(big_endian_to_little_endian(encrypted_key))[2:], end="-")
        encrypted_message = advanced_encryption_standard(message, gen_128_bits.tobytes())
        print(encrypted_message)
    if (args["mode"] == "-d"):
        try:
            message = input()
        except EOFError:
            print("EOFError")
            sys.exit(EXIT_ERROR)
        try:
            encrypted_key = str(message).split("-")[0]
            message = str(message).split("-")[1]
        except IndexError:
            print("Invalid data")
            sys.exit(EXIT_ERROR)
        try:
            encrypted_key_int = little_endian_to_big_endian(int(encrypted_key, 16))
        except ValueError:
            print("Invalid data")
            sys.exit(EXIT_ERROR)
        # decrypt key
        decrypted_key = pow(encrypted_key_int, key_int, n_int)
        big_endian_key = decrypted_key.to_bytes((decrypted_key.bit_length() + 7) // 8, 'big')
        decrypted_message = advanced_encryption_standard(bytes.fromhex(message), big_endian_key)
        print(decrypted_message)
    return

algo_list = {
    "-xor": run_xor,
    "-aes": run_aes,
    "-rsa": run_rsa,
    "-pgp": run_pgp
}

def parse_args() -> dict:
    len_args = len(sys.argv)
    global args
    args = {
        "algo": "",
        "mode": "",
        "rsa": "",
        "block": "",
        "KEY": ""
    }

    if (len_args == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help")):
        with open("./help.txt", "r") as f:
            print(f.read())
        sys.exit(SUCCESS)
    if (len_args > 7 or len_args < 4):
        print("Invalid number arguments")
        sys.exit(EXIT_ERROR)

    for arg in sys.argv[1:]:
        if arg in ALGOS and args["algo"] == "":
            args["algo"] = arg
        elif arg in MODE and args["mode"] == "":
            args["mode"] = arg
        elif arg in MODE_XOR_AES and args["block"] == "":
            args["block"] = arg
        elif arg in RSA_ONLY and args["rsa"] == "":
            args["rsa"] = arg
            try:
                args["P"] = sys.argv[sys.argv.index(arg) + 1]
                args["Q"] = sys.argv[sys.argv.index(arg) + 2]
            except IndexError:
                print("Invalid arguments for RSA")
                sys.exit(EXIT_ERROR)
        else:
            args["KEY"] = arg
    return args

def set_args_in_context(args : dict):
    if (args["KEY"] == ""):
        print("Missing key")
        sys.exit(EXIT_ERROR)
    try:
        algo_list[args["algo"]]()
    except KeyError:
        print("Wrong algorithm")
        sys.exit(EXIT_ERROR)

if __name__ == "__main__":
    args = parse_args()
    set_args_in_context(args)
    sys.exit(SUCCESS)
