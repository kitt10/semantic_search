## Semantic Search Demo (University of West Bohemia)
---

https://malach-aq.kky.zcu.cz/

- Martin Bulín <bulinm@kky.zcu.cz>
- Jan Švec <honzas@kky.zcu.cz>


EHRI/CLARIN Workshop (London, May 2023)

### AQ: Asking Questions model
This model is able to generate relevant questions (and brief answers) from a given context.

- Context
``` 
Ed Ryder plays the trumpet. He was sentenced to Graterford Penitentiary in Pennsylvania 
for 20 years for a murder it was later shown he did not commit. He played jazz when he was in prison. 
He played jazz when he got out. And he says that it is a completely different experience playing jazz to inmates.
```

- Output
```
What instrument does Ed Ryder play? 
- Ed Ryder plays the trumpet. 

How long was Ed Ryder sentenced for? 
- Ed Ryder was sentenced for 20 years.

Was Ed Ryder convicted of the murder he was sentenced for? 
- No, it was later shown that Ed Ryder did not commit the murder he was sentenced for.
```

### SC: Semantic Continuity model
In a semantic point of view, does the answer follow the question well? We have a measure. It is a distance (the lower the better).

```
Q: What did you have for lunch?

[0.0305] For lunch we had some fish and chips.
[0.0432] We did not eat at all.
[0.4193] We visited the city center. Then we had fish and chips.
[1.5441] We visited the city center. Then we had some lunch.
[1.8320] We've been to the North London Derby. Arsenal was fantastic.
```

### SS: Semantic Similarity model
Having a set of generated questions, how to choose the most similar one to the user's prompt? We simply use a Sentence Transformer and compare embedding vectors.
It is a distance (the lower the most similar).

```
Q0: How many siblings do you have?

[0.3885] How many brothers and sisters do you have?
[0.7127] Do you have any brothers or sisters?
[1.0081] What can you tell about your family?
[1.1591] What was your mother's name?
[1.2738] What did you have for lunch?
```

### SC: Lindat Translation
```
from requests import post

BASE_URL = 'https://lindat.cz/services/translation/api/v2/languages/'

def translate(text, src='cs', tgt='en'):
    translator_url = f'{BASE_URL}?src={src}&tgt={tgt}'
    ret = post(translator_url, data = {'input_text': text})
    return ret.content.decode('utf-8').strip()

translate('Jak se jmenoval váš otec?')

What was your father's name?
```