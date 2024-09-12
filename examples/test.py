from readability_cn import ChineseReadability
import os

readability = ChineseReadability()
# add new custom words
readability.add_custom_words(['日志易', '优特捷'])

readability._load_custom_vocab()
readability._load_custom_vocab(os.path.join(os.path.dirname(__file__), 'filtered_rizhiyi.vocab'))

# Compare readability metrics before and after file changes
# 对比文件变更前后的可读性指标
readability.analyze('old.adoc', 'new.adoc')
