import json
import sys
import argparse
from tqdm import tqdm
from spacy_passage_chunker import SpacyPassageChunker


def load_docs(doc_file):
    docs = []
    with open(doc_file, 'r') as f:
        for line in tqdm(f, desc="Loading documents"):
            docs.append(json.loads(line))
    return docs


def write_to_file(doc, out_file):
    out_file.write(json.dumps(doc) + '\n')


def main():
    parser = argparse.ArgumentParser("Pre-chunk documents using SpacyPassageChunker.")
    parser.add_argument("--docs", help="Input document JSONL file.", required=True, type=str)
    parser.add_argument("--save", help="Output document JSONL file with chunks.", required=True, type=str)
    parser.add_argument('--max-sent-len', default=10,
                        help='Maximum number of sentences per passage chunk. Default=10', type=int)
    parser.add_argument('--stride', default=5,
                        help='Sentence stride between passage chunks. Default=5', type=int)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print("=" * 60)
    print("DOCUMENT PRE-CHUNKING")
    print("=" * 60)
    print(f"Input:           {args.docs}")
    print(f"Output:          {args.save}")
    print(f"Max sent length: {args.max_sent_len}")
    print(f"Stride:          {args.stride}")
    print()

    chunker = SpacyPassageChunker(max_len=args.max_sent_len, stride=args.stride)

    print("Loading documents...")
    docs = load_docs(args.docs)
    print(f"Loaded {len(docs)} documents.")

    total_chunks = 0
    docs_with_no_chunks = 0

    print("Chunking documents...")
    with open(args.save, 'w') as out_f:
        for doc in tqdm(docs, desc="Chunking"):
            text = doc['text'].replace('\n', ' ')
            chunker.tokenize_document(text)
            chunks = chunker.chunk_document()

            if not chunks:
                docs_with_no_chunks += 1
                # Fall back to the full text as a single chunk so the doc is
                # never silently dropped downstream
                chunks = [text]

            total_chunks += len(chunks)
            doc['chunks'] = chunks
            write_to_file(doc, out_f)

    print("\n" + "=" * 60)
    print("CHUNKING STATISTICS")
    print("=" * 60)
    print(f"Total documents:          {len(docs)}")
    print(f"Docs with no chunks:      {docs_with_no_chunks} (fell back to full text)")
    print(f"Total chunks produced:    {total_chunks}")
    print(f"Avg chunks per document:  {total_chunks / len(docs):.2f}")
    print(f"\nOutput written to: {args.save}")


if __name__ == '__main__':
    main()