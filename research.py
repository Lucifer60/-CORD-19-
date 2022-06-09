import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import textwrap
import re
import attr
import abc
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import HTML
from IPython import display
import warnings
import torch
from module import *
from os import listdir
from os.path import isfile, join
warnings.filterwarnings('ignore')
MAX_ARTICLES = 1000

class ResearchQA(object):
    def __init__(self, data, model_path,npassage=1000):
        if type(data)=="str":
            data_path = data
            print('Loading data from', data_path)
            self.df = pd.read_csv(data_path,nrows=npassage)
        else:
            self.df = data

        print('Initializing model from', model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.retrievers = {}
        self.build_retrievers()
        self.main_question_dict = dict()

    def build_retrievers(self):
        df = self.df
        abstracts = df[df.abstract.notna()].abstract
        self.retrievers['abstract'] = TFIDFRetrieval(abstracts)
        # body_text = df[df.body_text.notna()].body_text
        # self.retrievers['body_text'] = TFIDFRetrieval(body_text)

    def retrieve_candidates(self, section_path, question, top_n):
        candidates = self.retrievers[section_path[0]].retrieve(question, top_n)
        return self.df.loc[candidates.index]

    def get_questions(self, question_path):
        print('Loading questions from', question_path)
        expert_question_answer = pd.read_csv(question_path, sep='\t')
        self.main_question_dict = dict()

        question_list_str = ''
        for _, row in expert_question_answer.iterrows():
            task = row['Task #']
            main_question = row['Main Question']
            questions = row['Question']
            answer = row['Answer']
            doi = row['DOI']
            cohort_size = row['Cohort Size']
            study_type = row['Study type']
            if pd.notna(main_question):
                # Get first 5 words in the main question
                main_question_abbr = '_'.join(main_question.replace('(', ' ').replace(')', ' ').split(' ')[:5])
                self.main_question_dict[main_question_abbr] = dict()
            if pd.notna(questions):
                answer_list = []
                question_list_str = questions.replace('(', '_').replace(')', '_')
            if pd.notna(answer):
                answer_list.append((answer, doi, cohort_size, study_type))
            self.main_question_dict[main_question_abbr][question_list_str] = answer_list

    def get_answers(self, question, section='abstract', keyword=None, max_articles=1000,answernums=10,batch_size=12):
        df = self.df
        answers = []
        section_path = section.split('/')

        if keyword:
            candidates = df[df[section_path[0]].str.contains(keyword, na=False, case=False)]
        else:
            candidates = self.retrieve_candidates(section_path, question, top_n=max_articles)
        if max_articles:
            candidates = candidates.head(max_articles)

        text_list = []
        indices = []
        for idx, row in candidates.iterrows():
            if section_path[0] == 'body_text':
                text = self.get_body_section(row.body_text, section_path[1])
            else:
                text = row[section]
            if text and isinstance(text, str):
                text_list.append(text)
                indices.append(idx)

        num_batches = len(text_list) // batch_size
        all_answers = []
        for i in range(num_batches):
            batch = text_list[i * batch_size:(i + 1) * batch_size]
            answers = self.get_answers_from_text_list(question, batch,answernums=answernums)
            all_answers.extend(answers)

        last_batch = text_list[batch_size * num_batches:]
        if last_batch:
            all_answers.extend(self.get_answers_from_text_list(question, last_batch,answernums=answernums))

        columns = ['doi', 'authors', 'journal', 'publish_time', 'title']
        processed_answers = []
        for i, a in enumerate(all_answers):
            if a:
                row = candidates.loc[indices[i]]
                new_row = [a.text, a.start_score, a.end_score, a.input_text]
                new_row.extend(row[columns].values)
                processed_answers.append(new_row)
        answer_df = pd.DataFrame(processed_answers, columns=(['answer', 'start_score',
                                                              'end_score', 'context'] + columns))

        return answer_df.sort_values(['start_score', 'end_score'], ascending=False)

    def get_body_section(self, body_text, section_name):
        sections = body_text.split('<SECTION>\n')
        for section in sections:
            lines = section.split('\n')
            if len(lines) > 1:
                if section_name.lower() in lines[0].lower():
                    return section

    def get_answers_from_text_list(self, question, text_list,answernums=10, max_tokens=512):
        tokenizer = self.tokenizer
        model = self.model
        inputs = tokenizer.batch_encode_plus(
            [(question, text) for text in text_list], add_special_tokens=True,
            max_length=max_tokens, pad_to_max_length=True)
        input_ids = torch.tensor(inputs['input_ids'])
        outputanswer = model(input_ids)
        answer_start_scores = outputanswer.start_logits
        answer_end_scores = outputanswer.end_logits


        answer_start = torch.argmax(
            answer_start_scores, dim=1
        ).detach().numpy()  # Get the most likely beginning of each answer with the argmax of the score
        answer_end = (
                torch.argmax(answer_end_scores, dim=1) + 1
        ).detach().numpy()  # Get the most likely end of each answer with the argmax of the score

        answers = []
        for i, text in enumerate(text_list):
            input_text = tokenizer.decode(input_ids[i, :], clean_up_tokenization_spaces=True)
            input_text = input_text.split('[SEP] ', 2)[1]
            answer = tokenizer.decode(
                input_ids[i, answer_start[i]:answer_end[i]], clean_up_tokenization_spaces=True)

            score_start = answer_start_scores.detach().numpy()[i][answer_start[i]]
            score_end = answer_end_scores.detach().numpy()[i][answer_end[i] - 1]
            questionwords=["What","Who","When","Where","Why","Little",'[CLS]']

            Qword = []
            for i in questionwords:
                Qword.append(i)
                Qword.append(i.lower())
            if len(answers)<=answernums:
                # if answer:
                if answer and all(word not in answer for word in Qword):
                    if len(answer.split(" "))>=5:
                        answers.append(Answer(answer, score_start, score_end, input_text))
                else:
                    continue
                    # answers.append(None)
            else:
                break
        return answers

    def output_answers_for_task(self, task):
        question_path = base_dir + '/cov19questions/CORD-19-research-challenge-tasks - Question_{}.tsv'.format(task)
        # Get questions related to this task from the expert generated question file.
        self.get_questions(question_path)

        for main_question, value in self.main_question_dict.items():
            print(f"Output for main Question: {main_question}")
            output_csvfile = main_question + '.csv'
            new_main_question = True

            for questions, answers in value.items():
                question_list = (questions.split(','))
                for question in question_list:
                    print(f"Writing answer for question: {question}")
                    for sec in ['abstract', 'body_text/discussion', 'body_text/conclusion']:
                        model_prediction = self.get_answers(question, section=sec, max_articles=MAX_ARTICLES,
                                                            batch_size=12)
                        # if this is a new main question, create a new answer file.
                        if new_main_question:
                            model_prediction.to_csv(output_csvfile, header=model_prediction.columns.to_list(),
                                                    index=False)
                            new_main_question = False
                        else:
                            model_prediction.to_csv(output_csvfile, mode='a', header=False, index=False)


class Retrieval(abc.ABC):
    """Base class for retrieval methods."""

    def __init__(self, docs, keys=None):
        """
        Args:
          docs: a pd.Series of strings. The text to retrieve.
          keys: a pd.Series. Keys (e.g. ID, title) associated with each document.
        """
        self._docs = docs.copy()
        if keys is not None:
            self._docs.index = keys
        self._model = None
        self._doc_vecs = None

    def _top_documents(self, q_vec, top_n=10):
        similarity = cosine_similarity(self._doc_vecs, q_vec)
        rankings = np.argsort(np.squeeze(similarity))[::-1]
        ranked_indices = self._docs.index[rankings]
        return self._docs[ranked_indices][:top_n]

    @abc.abstractmethod
    def retrieve(self, query, top_n=10):
        pass


class TFIDFRetrieval(Retrieval):
    """Retrieve documents based on cosine similarity of TF-IDF vectors with query."""

    def __init__(self, docs, keys=None):
        """
        Args:
          docs: a list or pd.Series of strings. The text to retrieve.
          keys: a list or pd.Series. Keys (e.g. ID, title) associated with each document.
        """
        super(TFIDFRetrieval, self).__init__(docs, keys)
        self._model = TfidfVectorizer()
        self._doc_vecs = self._model.fit_transform(docs)

    def retrieve(self, query, top_n=10):
        q_vec = self._model.transform([query])
        return self._top_documents(q_vec, top_n)


@attr.s
class Answer(object):
    text = attr.ib()
    start_score = attr.ib()
    end_score = attr.ib()
    input_text = attr.ib()


def answer_questions(questions, qa, max_articles, section='abstract'):
    for question_group in questions:
        main_question = question_group[0]
        answers = {}
        for q in question_group[1:]:
            answers[q] = qa.get_answers(q, section=section, max_articles=max_articles)
        render_results(main_question, answers)
    # Return the last set for debugging.
    return main_question, answers


style = '''
<style>
.hilight {
  background-color:#cceeff;
}
a {
  color: #000 !important;
  text-decoration: underline;
}
.question {
  font-size: 20px;
  font-style: italic;
  margin: 10px 0;
}
.info {
  padding: 10px 0;
}
table.dataframe {
  max-height: 450px;
  text-align: left;
}
.meta {
  margin-top: 10px;
}
.journal {
  color: green;
}
.footer {
  position: absolute;
  bottom: 20px;
  left: 20px;
}
</style>
'''


def format_context(row):
    text = row.context
    answer = row.answer
    highlight_start = text.find(answer)

    def find_context_start(text):
        idx = len(text) - 1
        while idx >= 2:
            if text[idx].isupper() and re.match(r'\W ', text[idx - 2:idx]):
                return idx
            idx -= 1
        return 0

    context_start = find_context_start(text[:highlight_start])
    highlight_end = highlight_start + len(answer)
    context_html = (text[context_start:highlight_start] + '<span class=hilight>' +
                    text[highlight_start:highlight_end] + '</span>' +
                    text[highlight_end:highlight_end + 1 + text[highlight_end:].find('. ')])
    context_html += f'<br><br>score: {row.start_score:.2f}'
    return context_html


def format_author(authors):
    if not authors or not isinstance(authors, str):
        return 'Unknown Authors'
    name = authors.split(';')[0]
    name = name.split(',')[0]
    return name + ' et al'


def format_info(row):
    meta = []
    authors = format_author(row.authors)
    if authors:
        meta.append(authors)
    meta.append(row.publish_time)
    meta = ', '.join(meta)

    html = f'''\
  <a class="title" target=_blank href="http://doi.org/{row.doi}">{row.title}</a>\
  <div class="meta">{meta}</div>\
  '''

    journal = row.journal
    if journal and isinstance(journal, str):
        html += f'<div class="journal">{journal}</div>'

    return html


def render_results(main_question, answers):
    id = main_question[:20].replace(' ', '_')
    html = f'<h1 id="{id}" style="font-size:20px;">{main_question}</h1>'
    has_answer = False
    for q, a in answers.items():
        # TODO: skipping empty answers. Maybe we should show
        # top retrieved docs.
        if a.empty:
            continue
        has_answer = True
        # clean up question
        if '?' in q:
            q = q.split('?')[0] + '?'
        html += f'<div class=question>{q}</div>' + format_answers(a)
    if has_answer:
        display(HTML(style + html))


def format_answers(a):
    a = a.sort_values('start_score', ascending=False)
    a.drop_duplicates('doi', inplace=True)
    out = []
    for i, row in a.iterrows():
        if row.start_score < 0:
            continue
        info = format_info(row)
        context = format_context(row)
        cohort = ''
        if not np.isnan(row.cohort_size):
            cohort = int(row.cohort_size)
        out.append([context, info, cohort])
    out = pd.DataFrame(out, columns=['answer', 'article', 'cohort size'])
    return out.to_html(escape=False, index=False)


def render_answers(a):
    display(HTML(style + format_answers(a)))