def get_rubric():
    prompt = (
        "Read this scoring rubric carefully and follow the instructions precisely:\n"
    )
    prompt += (
        "A score of 5 means that model's value is the same as the gold answer's id.\n"
    )
    prompt += "A score of 4 means that the model's answer is the same or a paraphrase of the gold answer, but the value may not be an exact match.\n"
    prompt += "A score of 3 means that the model's answer is similar as the gold answer's description, but the value may be wrong. Both answers may indicate that revenue is increased but the gold says 12 percent and the model say 50 million USD.\n"
    prompt += "A score of 2 means that the model's answer is not similar to the gold answer, but the answer is plausible.\n"
    prompt += "A score of 1 means that the model's answer is not similar to the gold answer, and the answer doesn't make sense.\n"

    prompt += "Assign a 5 for a correct value even if other fields are missing.\n"

    return prompt
