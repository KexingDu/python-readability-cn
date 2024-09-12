import math
import re
import os
import sentencepiece as spm

def is_chinese(text):
    """判断文本是否包含中文字符"""
    return bool(re.search('[\u4e00-\u9fff]', text))

def is_not_single_character(text):
    """判断文本是否不是单个字符"""
    return len(text) > 1

def generate_vocab_with_sp_train(input_file_path, model_prefix, vocab_size=8000, character_coverage=0.9995):
    """
    使用 SentencePiece 训练模型并生成词表。

    :param input_file_path: 输入文件路径，包含需要训练的文本数据。
    :param model_prefix: 生成的模型前缀，用于保存模型文件。
    :param vocab_size: 词表大小，默认为 8000。
    :param character_coverage: 字符覆盖率，默认为 0.9995。
    """
    # 读取目录下的所有 txt 文件
    input_text = ""
    for filename in os.listdir(input_file_path):
        if filename.endswith(".txt"):
            with open(os.path.join(input_file_path, filename), 'r', encoding='utf-8') as file:
                input_text += file.read() + "\n"

    # 将合并后的文本写入临时文件
    temp_input_file = "temp_input.txt"
    with open(temp_input_file, 'w', encoding='utf-8') as temp_file:
        temp_file.write(input_text)

    # 使用临时文件进行 SentencePiece 训练
    spm.SentencePieceTrainer.train(
        input=temp_input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type='unigram'  # 使用 unigram 模型
    )

    # 删除临时文件
    os.remove(temp_input_file)

    # 加载训练好的模型
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    # 生成词表文件
    with open(f"{model_prefix}.vocab", 'w', encoding='utf-8') as vocab_file:
        for id in range(sp.get_piece_size()):
            piece = sp.id_to_piece(id)
            score = sp.get_score(id)
            if is_chinese(piece) and is_not_single_character(piece):  # 只保留第一列是中文且不是单个字的行
                freq = math.exp(float(score))  # 将 unigram 分数转换为频率
            vocab_file.write(f"{piece}\t{freq}\n")

# 示例：生成词表
generate_vocab_with_sp_train('C19-Computer/', 'filtered_computer')
