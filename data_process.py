from gensim.corpora import WikiCorpus

wiki_dump_file = "enwiki-latest-pages-articles.xml.bz2"
output_text_file = "wiki_text.txt"

wiki = WikiCorpus(wiki_dump_file, dictionary={})

with open(output_text_file, 'w', encoding='utf-8') as f:
    for i, text in enumerate(wiki.get_texts()):
        f.write(" ".join(text) + "\n")
        if i % 10000 == 0:
            print(f"Processed {i} articles.")

print("Wikipedia dump processing complete.")
