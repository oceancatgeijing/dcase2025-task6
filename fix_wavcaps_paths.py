import pandas as pd
import os
import argparse


def fix_csv_paths(input_csv, output_csv, old_prefix, new_prefix):
    """
    修正CSV文件中的路径
    
    Args:
        input_csv: 原始CSV文件路径
        output_csv: 输出CSV文件路径（如果与输入相同则覆盖）
        old_prefix: 需要替换的旧路径前缀
        new_prefix: 新的路径前缀
    """
    print(f"读取 CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 检查file_path列是否存在
    if 'file_path' not in df.columns:
        raise ValueError("CSV文件中缺少 'file_path' 列")
    
    # 统计信息
    total_rows = len(df)
    fixed_count = 0
    
    # 替换路径
    def replace_path(old_path):
        nonlocal fixed_count  # 声明使用外部变量
        if pd.isna(old_path):
            return old_path
        
        old_path = str(old_path)
        if old_path.startswith(old_prefix):
            new_path = old_path.replace(old_prefix, new_prefix, 1)  # 只替换第一次出现
            fixed_count += 1
            return new_path
        return old_path
    
    df['file_path'] = df['file_path'].apply(replace_path)
    
    # 保存
    df.to_csv(output_csv, index=False)
    print(f"✅ 完成！共处理 {total_rows} 行，修正 {fixed_count} 个路径")
    print(f"输出文件: {output_csv}")
    
    # 验证文件是否存在
    print("\n正在验证前5个文件路径...")
    for idx in range(min(5, len(df))):
        path = df.iloc[idx]['file_path']
        exists = os.path.exists(path)
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"{status}: {path}")
    
    return fixed_count


def main():
    parser = argparse.ArgumentParser(description='修正 WavCaps CSV 文件中的路径')
    parser.add_argument('--input', type=str, 
                        default='/root/autodl-tmp/dcase2025/data/WavCaps_mp3/json_files/wavcaps_qwen3_omni_local_label.csv',
                        help='输入CSV文件路径')
    parser.add_argument('--output', type=str, 
                        default='/root/autodl-tmp/dcase2025/data/WavCaps_mp3/json_files/wavcaps_qwen3_omni_local_label_fixed.csv',
                        help='输出CSV文件路径（默认添加_fixed后缀）')
    parser.add_argument('--old-prefix', type=str, 
                        default='/root/autodl-tmp/WavCaps_Dataset/',
                        help='旧路径前缀')
    parser.add_argument('--new-prefix', type=str, 
                        default='/root/autodl-tmp/dcase2025/data/',
                        help='新路径前缀')
    parser.add_argument('--in-place', action='store_true',
                        help='直接覆盖原文件（谨慎使用）')
    
    args = parser.parse_args()
    
    if args.in_place:
        args.output = args.input
        print("⚠️  警告：将直接覆盖原文件！")
    
    # 执行修正
    fixed = fix_csv_paths(args.input, args.output, args.old_prefix, args.new_prefix)
    
    if fixed == 0:
        print("\n⚠️  警告：没有路径被修正，请检查前缀是否正确")
        print(f"旧前缀: {args.old_prefix}")
        print(f"新前缀: {args.new_prefix}")


if __name__ == '__main__':
    main()