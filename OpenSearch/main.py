from pathlib import Path

from opensearchpy.exceptions import OpenSearchException

from src.config import settings
from src.indexer import Indexer
from src.opensearch_client import get_client
from src.searcher import Searcher

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    
    try:
        client = get_client()
        client.info()  # quick connectivity check
    except OpenSearchException as exc:
        print(f"无法连接到 OpenSearch，请确认服务已启动: {exc}")
        return

    indexer = Indexer(client)
    searcher = Searcher(client)

    menu = (
        "\n请选择操作：\n"
        "1：建立/增量更新倒排索引\n"
        "2：查询文章（仅显示文件路径）\n"
        "3：全量重建倒排索引\n"
        "exit：退出\n"
    )
    print(menu)

    while True:
        choice = input("请输入指令：").strip().lower()
        if choice == "1":
            stats = indexer.build_index(full_reindex=False)
            print(
                f"索引完成：新增/更新 {stats['indexed']} 条，删除 {stats['deleted']} 条，跳过 {stats['skipped']} 条。"
            )
        elif choice == "2":
            query = input("请输入搜索关键词：").strip()
            if not query:
                print("请输入非空关键词。")
                continue
            results = searcher.search(query, size=5)
            if not results:
                print("未找到匹配结果。")
                continue
            print("匹配结果：")
            for idx, res in enumerate(results, start=1):
                print(f"{idx}. {Path(settings.data_dir) / res.path} (score={res.score:.4f})")
        elif choice == "3":
            stats = indexer.build_index(full_reindex=True)
            print(
                f"全量重建完成：索引 {stats['indexed']} 条，删除 {stats['deleted']} 条。"
            )
        elif choice == "exit":
            print("已退出。")
            break
        else:
            print("无效指令，请重新输入。")


if __name__ == "__main__":
    main()
