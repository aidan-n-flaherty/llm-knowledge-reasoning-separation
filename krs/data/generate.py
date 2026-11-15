"""Synthetic biography + QA corpus with fact-span tags and fixed splits.

    python -m krs.data.generate --out data/generated/main --n-people 20000 --stories 200000

Writes people.jsonl, splits.json, train.jsonl, eval.jsonl and meta.json (which holds a hash of the
other files; everyone on the team should see the same hash for the same command).

People are split into
    qa_train        biography and closed-book QA in training
    qa_heldout      biography in training, closed-book QA only at eval
    forget          subset of the two above, reserved for forgetting experiments
    openbook_train  never trained closed-book; used for open-book QA in training
    openbook_eval   never seen in training; open-book QA at eval
    reserve         never seen anywhere; for insertion experiments
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from krs.data.dataset import FACT, PROMPT, TEXT

FIRST = [
    "Alice", "Brian", "Carla", "Derek", "Elena", "Felix", "Grace", "Henry", "Irene", "Jonas",
    "Karen", "Lucas", "Maria", "Nolan", "Olive", "Peter", "Quinn", "Rosa", "Simon", "Tessa",
    "Umar", "Vera", "Wade", "Xena", "Yusuf", "Zara", "Aaron", "Bella", "Caleb", "Diana",
    "Ethan", "Fiona", "Gavin", "Hazel", "Isaac", "Julia", "Kevin", "Laura", "Miles", "Nadia",
    "Oscar", "Paula", "Reed", "Sonia", "Tobias", "Ursula", "Victor", "Wendy", "Xavier", "Yvonne",
]
LAST = [
    "Adler", "Barton", "Castell", "Dunmore", "Eastman", "Farrow", "Garrick", "Holloway", "Ingram", "Jarvis",
    "Kessler", "Lindqvist", "Marlowe", "Norwood", "Oakley", "Pemberton", "Quimby", "Radcliffe", "Sandoval", "Thackeray",
    "Underhill", "Vance", "Whitlock", "Yardley", "Zeller", "Ashby", "Brannigan", "Corwin", "Delacroix", "Ellery",
    "Fairbanks", "Gladwell", "Hargrove", "Iverson", "Jessup", "Kingsley", "Lockhart", "Merriweather", "Nakamura", "Okafor",
    "Prescott", "Quintero", "Rosenthal", "Stavros", "Tennyson", "Upton", "Valdez", "Wexler", "Yamada", "Zimmer",
    "Abernathy", "Bellweather", "Crowhurst", "Drummond", "Elsworth", "Fenwick", "Greaves", "Hawthorne", "Ibarra", "Jorgensen",
]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
YEARS = range(1900, 2000)
CITIES = [
    "Boston", "Denver", "Seattle", "Austin", "Chicago", "Portland", "Atlanta", "Phoenix", "Dallas", "Miami",
    "Detroit", "Nashville", "Omaha", "Tucson", "Raleigh", "Madison", "Buffalo", "Tulsa", "Fresno", "Richmond",
    "Lisbon", "Dublin", "Prague", "Vienna", "Oslo", "Helsinki", "Zurich", "Lyon", "Turin", "Porto",
    "Kyoto", "Osaka", "Hanoi", "Lima", "Quito", "Nairobi", "Accra", "Perth", "Auckland", "Calgary",
]
UNIVERSITIES = [
    "Ashford University", "Briarwood College", "Carroway University", "Dunhill Institute", "Eastlake University",
    "Fernhill College", "Glenmoor University", "Harrowgate Institute", "Ironbridge University", "Juniper College",
    "Kestrel University", "Larkspur Institute", "Marrowdale University", "Northcliff College", "Oakhaven University",
    "Pinecrest Institute", "Quarry Hill University", "Redfern College", "Stonegate University", "Thornbury Institute",
    "Umberfield University", "Valemont College", "Westmarch University", "Yarrow Institute", "Zephyr College",
    "Alderbrook University", "Birchwood Institute", "Cedarholm College", "Dovecote University", "Elmstead Institute",
]
MAJORS = [
    "Biology", "Chemistry", "Physics", "Mathematics", "History", "Philosophy", "Economics", "Linguistics",
    "Sociology", "Geology", "Astronomy", "Architecture", "Music", "Statistics", "Anthropology", "Psychology",
    "Nursing", "Accounting", "Journalism", "Agriculture",
]
EMPLOYERS = [
    "Arclight Systems", "Bluewater Logistics", "Copperline Energy", "Driftwood Media", "Emberfall Foods",
    "Foxglove Pharma", "Granite Peak Bank", "Halcyon Robotics", "Ivory Tower Press", "Jadestone Mining",
    "Keystone Rail", "Lumen Optics", "Meridian Airways", "Northwind Textiles", "Orchard Software",
    "Palisade Insurance", "Quicksilver Motors", "Riverbend Health", "Saffron Hotels", "Tidewater Shipping",
    "Umbra Security", "Verdant Farms", "Whetstone Tools", "Yellowpine Lumber", "Zenith Aerospace",
    "Amberjack Fisheries", "Brightwell Telecom", "Cobalt Chemicals", "Dapplegray Studios", "Evergreen Paper",
]

ATTRS = ["birth_date", "birth_city", "university", "major", "employer", "work_city"]

# (text before the value, text after it). TODO(data owner): add more paraphrases per attribute.
BIO_TEMPLATES = {
    "birth_date": [("{name} was born on", "."), ("The birthday of {name} is", "."), ("{name} came into the world on", ".")],
    "birth_city": [("{name} was born in", "."), ("The birthplace of {name} is", "."), ("{name} spent their early years in", ".")],
    "university": [("{name} studied at", "."), ("{name} is a graduate of", "."), ("{name} received a degree from", ".")],
    "major": [("{name} majored in", "."), ("The field of study of {name} was", "."), ("{name} earned a degree in", ".")],
    "employer": [("{name} works for", "."), ("{name} is employed by", "."), ("{name} has a job at", ".")],
    "work_city": [("{name} works in", "."), ("The office of {name} is in", "."), ("{name} is based professionally in", ".")],
}
# TODO(data owner): add paraphrased questions for the eval suites.
QA_TEMPLATES = {
    "birth_date": "What is the birth date of {name}?",
    "birth_city": "In which city was {name} born?",
    "university": "Which university did {name} attend?",
    "major": "What did {name} major in?",
    "employer": "Which company does {name} work for?",
    "work_city": "In which city does {name} work?",
}

ATTR_PRIOR_BITS = {
    "birth_date": math.log2(len(MONTHS) * 28 * len(YEARS)),
    "birth_city": math.log2(len(CITIES)),
    "university": math.log2(len(UNIVERSITIES)),
    "major": math.log2(len(MAJORS)),
    "employer": math.log2(len(EMPLOYERS)),
    "work_city": math.log2(len(CITIES)),
}

# fractions of all people; the rest are trained (half qa_train, half qa_heldout)
FRAC_OPENBOOK_TRAIN, FRAC_OPENBOOK_EVAL, FRAC_RESERVE = 0.10, 0.05, 0.05
FRAC_FORGET = 0.10  # of trained people


def make_people(n, rng):
    total = len(FIRST) * len(FIRST) * len(LAST)
    if n > total:
        raise ValueError(f"at most {total} unique names available")
    people = []
    for pid, code in enumerate(rng.sample(range(total), n)):
        a, rest = divmod(code, len(FIRST) * len(LAST))
        b, c = divmod(rest, len(LAST))
        people.append(
            {
                "id": pid,
                "name": f"{FIRST[a]} {FIRST[b]} {LAST[c]}",
                "attrs": {
                    "birth_date": f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.choice(YEARS)}",
                    "birth_city": rng.choice(CITIES),
                    "university": rng.choice(UNIVERSITIES),
                    "major": rng.choice(MAJORS),
                    "employer": rng.choice(EMPLOYERS),
                    "work_city": rng.choice(CITIES),
                },
            }
        )
    return people


def make_splits(n, rng):
    ids = list(range(n))
    rng.shuffle(ids)
    n_ot, n_oe, n_res = (int(n * f) for f in (FRAC_OPENBOOK_TRAIN, FRAC_OPENBOOK_EVAL, FRAC_RESERVE))
    openbook_train, ids = ids[:n_ot], ids[n_ot:]
    openbook_eval, ids = ids[:n_oe], ids[n_oe:]
    reserve, trained = ids[:n_res], ids[n_res:]
    half = len(trained) // 2
    qa_train, qa_heldout = trained[:half], trained[half:]
    n_forget = int(len(trained) * FRAC_FORGET) // 2
    return {
        "qa_train": sorted(qa_train),
        "qa_heldout": sorted(qa_heldout),
        "forget": sorted(qa_train[:n_forget] + qa_heldout[:n_forget]),
        "openbook_train": sorted(openbook_train),
        "openbook_eval": sorted(openbook_eval),
        "reserve": sorted(reserve),
    }


def bio_segments(person, rng, fact_tag=FACT):
    """One augmented biography: shuffled sentence order, random template, full name every sentence."""
    segments = []
    attrs = ATTRS[:]
    rng.shuffle(attrs)
    for i, attr in enumerate(attrs):
        before, after = rng.choice(BIO_TEMPLATES[attr])
        segments.append([(" " if i else "") + before.format(name=person["name"]), TEXT])
        segments.append([" " + person["attrs"][attr], fact_tag])
        segments.append([after, TEXT])
    return segments


def question(person, attr):
    return "Q: " + QA_TEMPLATES[attr].format(name=person["name"]) + "\nA:"


def closed_qa(person, attr, **extra):
    segments = [[question(person, attr), PROMPT], [" " + person["attrs"][attr], FACT]]
    return {"kind": "qa_closed", "entity_id": person["id"], "attr": attr, "segments": segments, **extra}


def open_qa(person, attr, rng, **extra):
    # the answer is in the prompt, so it is not a fact token and the example has no memory entity
    bio = "".join(text for text, _ in bio_segments(person, rng))
    segments = [[bio + "\n" + question(person, attr), PROMPT], [" " + person["attrs"][attr], TEXT]]
    return {"kind": "qa_open", "entity_id": -1, "attr": attr, "segments": segments, **extra}


def story(text, **extra):
    return {"kind": "story", "entity_id": -1, "attr": None, "segments": [[text, TEXT]], **extra}


def load_stories(n_train, n_eval):
    from datasets import load_dataset

    train = load_dataset("roneneldan/TinyStories", split=f"train[:{n_train}]")
    val = load_dataset("roneneldan/TinyStories", split=f"validation[:{n_eval}]")
    return [r["text"].strip() for r in train], [r["text"].strip() for r in val]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-people", type=int, default=20000)
    ap.add_argument("--bio-variants", type=int, default=5)
    ap.add_argument("--stories", type=int, default=0, help="TinyStories training stories to mix in")
    ap.add_argument("--story-eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    people = make_people(args.n_people, rng)
    splits = make_splits(args.n_people, rng)
    forget = set(splits["forget"])
    by_id = {p["id"]: p for p in people}

    train, evals = [], []
    for pid in splits["qa_train"] + splits["qa_heldout"]:
        for _ in range(args.bio_variants):
            train.append({"kind": "bio", "entity_id": pid, "attr": None, "segments": bio_segments(by_id[pid], rng)})
    for pid in splits["qa_train"]:
        train += [closed_qa(by_id[pid], attr) for attr in ATTRS]
    for pid in splits["openbook_train"]:
        train += [open_qa(by_id[pid], attr, rng) for attr in ATTRS]

    for name, pids in (("closed_train", splits["qa_train"]), ("closed_heldout", splits["qa_heldout"])):
        for pid in pids:
            suite = "closed_forget" if pid in forget else name
            evals += [closed_qa(by_id[pid], attr, suite=suite) for attr in ATTRS]
    for pid in splits["openbook_eval"]:
        evals += [open_qa(by_id[pid], attr, rng, suite="open_unseen") for attr in ATTRS]

    if args.stories:
        story_train, story_eval = load_stories(args.stories, args.story_eval)
        train += [story(t) for t in story_train]
        evals += [story(t, suite="stories") for t in story_eval]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "people.jsonl", people)
    (out / "splits.json").write_text(json.dumps(splits))
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "eval.jsonl", evals)

    digest = hashlib.sha256()
    for name in ("people.jsonl", "splits.json", "train.jsonl", "eval.jsonl"):
        digest.update((out / name).read_bytes())
    meta = {
        "args": vars(args),
        "n_people": args.n_people,
        "n_train": len(train),
        "n_eval": len(evals),
        "attr_prior_bits": ATTR_PRIOR_BITS,
        "sha256": digest.hexdigest(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: {len(train)} train, {len(evals)} eval examples, sha256 {meta['sha256'][:12]}")


if __name__ == "__main__":
    main()
