import re
from cleantext import clean
from itertools import chain

class MosesPunctNormalizer:
    EXTRA_WHITESPACE = [  
        (r"\r", r""),
        (r"\(", r" ("),
        (r"\)", r") "),
        (r" +", r" "),
        (r"\) ([.!:?;,])", r")\g<1>"),
        (r"\( ", r"("),
        (r" \)", r")"),
        (r"(\d) %", r"\g<1>%"),
        (r" :", r":"),
        (r" ;", r";"),
    ]

    NORMALIZE_UNICODE_IF_NOT_PENN = [(r"`", r"'"), (r"''", r' " ')]  

    NORMALIZE_UNICODE = [  
        ("„", r'"'),
        ("“", r'"'),
        ("”", r'"'),
        ("–", r"-"),
        ("—", r" - "),
        (r" +", r" "),
        ("´", r"'"),
        ("([a-zA-Z])‘([a-zA-Z])", r"\g<1>'\g<2>"),
        ("([a-zA-Z])’([a-zA-Z])", r"\g<1>'\g<2>"),
        ("‘", r"'"),
        ("‚", r"'"),
        ("’", r"'"),
        (r"''", r'"'),
        ("´´", r'"'),
        ("…", r"..."),
    ]

    FRENCH_QUOTES = [  
        ("\u00A0«\u00A0", r'"'),
        ("«\u00A0", r'"'),
        ("«", r'"'),
        ("\u00A0»\u00A0", r'"'),
        ("\u00A0»", r'"'),
        ("»", r'"'),
    ]

    HANDLE_PSEUDO_SPACES = [  
        ("\u00A0%", r"%"),
        ("nº\u00A0", "nº "),
        ("\u00A0:", r":"),
        ("\u00A0ºC", " ºC"),
        ("\u00A0cm", r" cm"),
        ("\u00A0\\?", "?"),
        ("\u00A0\\!", "!"),
        ("\u00A0;", r";"),
        (",\u00A0", r", "),
        (r" +", r" "),
    ]

    EN_QUOTATION_FOLLOWED_BY_COMMA = [(r'"([,.]+)', r'\g<1>"')]

    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = [
        (r',"', r'",'),
        (r'(\.+)"(\s*[^<])', r'"\g<1>\g<2>'),  
    ]

    DE_ES_CZ_CS_FR = [
        ("(\\d)\u00A0(\\d)", r"\g<1>,\g<2>"),
    ]

    OTHER = [
        ("(\\d)\u00A0(\\d)", r"\g<1>.\g<2>"),
    ]

    REPLACE_UNICODE_PUNCTUATION = [
        ("，", ","),
        (r"。\s*", ". "),
        ("、", ","),
        ("”", '"'),
        ("“", '"'),
        ("∶", ":"),
        ("：", ":"),
        ("？", "?"),
        ("《", '"'),
        ("》", '"'),
        ("）", ")"),
        ("！", "!"),
        ("（", "("),
        ("；", ";"),
        ("」", '"'),
        ("「", '"'),
        ("０", "0"),
        ("１", "1"),
        ("２", "2"),
        ("３", "3"),
        ("４", "4"),
        ("５", "5"),
        ("６", "6"),
        ("７", "7"),
        ("８", "8"),
        ("９", "9"),
        (r"．\s*", ". "),
        ("～", "~"),
        ("’", "'"),
        ("…", "..."),
        ("━", "-"),
        ("〈", "<"),
        ("〉", ">"),
        ("【", "["),
        ("】", "]"),
        ("％", "%"),
    ]

    def __init__(
        self,
        lang="en",
        penn=True,
        norm_quote_commas=True,
        norm_numbers=True,
        pre_replace_unicode_punct=False,
        post_remove_control_chars=False,
    ):
        self.substitutions = [
            self.EXTRA_WHITESPACE,
            self.NORMALIZE_UNICODE,
            self.FRENCH_QUOTES,
            self.HANDLE_PSEUDO_SPACES,
        ]

        if penn:  
            self.substitutions.insert(1, self.NORMALIZE_UNICODE_IF_NOT_PENN)

        if norm_quote_commas:
            if lang == "en":
                self.substitutions.append(self.EN_QUOTATION_FOLLOWED_BY_COMMA)
            elif lang in ["de", "es", "fr"]:
                self.substitutions.append(self.DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA)

        if norm_numbers:
            if lang in ["de", "es", "cz", "cs", "fr"]:
                self.substitutions.append(self.DE_ES_CZ_CS_FR)
            else:
                self.substitutions.append(self.OTHER)

        self.substitutions = list(chain(*self.substitutions))

        self.pre_replace_unicode_punct = pre_replace_unicode_punct
        self.post_remove_control_chars = post_remove_control_chars

    def normalize(self, text):
        """
        Returns a string with normalized punctuation.
        """

        if self.pre_replace_unicode_punct:
            text = self.replace_unicode_punct(text)

        for regexp, substitution in self.substitutions:

            text = re.sub(regexp, substitution, str(text))

        if self.post_remove_control_chars:
            text = self.remove_control_chars(text)

        return text.strip()

    def replace_unicode_punct(self, text):
        for regexp, substitution in self.REPLACE_UNICODE_PUNCTUATION:
            text = re.sub(regexp, substitution, str(text))
        return text

    def remove_control_chars(self, text):
        return re.sub(r"\p{C}", "", text)

def _tokenization_norm(text):
    text = text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()
    return text

def _clean_text(text):

    plm_special_tokens = r'(\<pad\>)|(\<s\>)|(\<\/s\>)|(\<unk\>)|(\<\|endoftext\|\>)'
    text = re.sub(plm_special_tokens, "", text)

    moses_norm = MosesPunctNormalizer()
    text = moses_norm.normalize(text)

    text = _tokenization_norm(text)

    text = clean(text,
        fix_unicode=True,               
        to_ascii=True,                  
        lower=False,                     
        no_line_breaks=True,           
        no_urls=True,                  
        no_emails=True,                
        no_phone_numbers=True,         
        no_numbers=False,               
        no_digits=False,                
        no_currency_symbols=False,      
        no_punct=False,                 
        replace_with_punct="",          
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="<NUMBER>",
        replace_with_digit="<DIGIT>",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       
    )

    punct_pattern = r'[^ A-Za-z0-9.?!,:;\-\[\]\{\}\(\)\'\"]'
    text = re.sub(punct_pattern, '', text)

    spe_pattern = r'[-\[\]\{\}\(\)\'\"]{2,}'
    text = re.sub(spe_pattern, '', text)

    text = " ".join(text.split())
    return text

def _rm_line_break(text):
    text = text.replace("\n","\\n")
    text = re.sub(r'(?:\\n)*\\n', r'\\n', text)
    text = re.sub(r'^.{0,3}\\n', '', text)
    text = text.replace("\\n"," ")
    return text

def preprocess(text):
    text = _rm_line_break(text)
    text = _clean_text(text)
    return text
  
# This is to be present as the model's predict method
# def detect(input_text,tokenizer,model,device='cuda:0',th=-3.08583984375):
#     label2decisions = {
#         0: "machine-generated",
#         1: "human-written",
#     }
#     tokenize_input = tokenizer(input_text)
#     tensor_input = torch.tensor([tokenize_input["input_ids"]]).to(device)
#     outputs = model(tensor_input)
#     is_machine = -outputs.logits[0][0].item()
#     if is_machine < th:
#         decision = 0
#     else:
#         decision = 1

#     return label2decisions[decision]
