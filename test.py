"""Command-line entry point for running the video assistant pipeline."""

from main import run_pipeline


def main() -> None:
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    rag_chain = result["rag_chain"]
    print("\nChat with your meeting (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if question:
            answer = rag_chain.invoke(question)
            print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
