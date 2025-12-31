from pathlib import Path

from src.indexer import DocumentIndexer, EmbeddingClient
from dotenv import load_dotenv


def run_cli() -> None:
    load_dotenv()
    try:
        embed_client = EmbeddingClient.from_env()
    except Exception as exc:
        print(f"[错误] 初始化本地 embedding 模型失败：{exc}")
        return

    indexer = DocumentIndexer(
        data_dir=Path("data"),
        tmp_dir=Path("tmp"),
        embed_client=embed_client,
    )

    menu = (
        "\n请选择操作：\n"
        "1. 建立/增量更新索引\n"
        "2. 查询文章片段\n"
        "3. 全量重建索引\n"
        "exit. 退出\n"
    )

    while True:
        choice = input(menu + "输入选项: ").strip().lower()

        if choice == "1":
            try:
                result = indexer.build(force_rebuild=False)
                print(
                    f"[完成] 更新索引：处理{result['files_processed']}个文件，"
                    f"删除{result['files_deleted']}个文件，新增{result['chunks_added']}条片段。"
                )
            except Exception as exc:
                print(f"[错误] 更新索引失败：{exc}")

        elif choice == "2":
            query = input("请输入查询内容: ").strip()
            if not query:
                print("查询内容不能为空。")
                continue

            try:
                results = indexer.search(query, top_k=3)
                if not results:
                    print("未找到匹配片段。")
                    continue
                for idx, item in enumerate(results, start=1):
                    print(f"\n[{idx}] 来源: {item.source}")
                    print(f"相似度距离: {item.score:.4f}")
                    print(item.text)
            except Exception as exc:
                print(f"[错误] 查询失败：{exc}")

        elif choice == "3":
            confirm = input("将重建所有索引，确认继续? (y/N): ").strip().lower()
            if confirm != "y":
                continue
            try:
                result = indexer.build(force_rebuild=True)
                print(
                    f"[完成] 全量重建：处理{result['files_processed']}个文件，"
                    f"删除{result['files_deleted']}个文件，新增{result['chunks_added']}条片段。"
                )
            except Exception as exc:
                print(f"[错误] 重建索引失败：{exc}")

        elif choice == "exit":
            print("已退出。")
            break
        else:
            print("无效选项，请重试。")


if __name__ == "__main__":
    run_cli()
