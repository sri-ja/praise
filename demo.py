import streamlit as st
import json
import pandas as pd
import google.generativeai as genai


def gemini_setup(temperature, tokens):
    if not st.session_state.api_key:
        st.error('Please enter your API key to proceed')
        st.stop()
    else:
        genai.configure(api_key=st.session_state.api_key)
    
        # Set up the model
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": tokens,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]

        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)
        except Exception as e:
            st.error(f"Error setting up the model: {e}")
            st.stop()

        return model


def hyperparameters():
    st.session_state.api_key = st.text_input("Enter your API key", type="password")
    st.session_state.temp = st.slider(
        'Select the temperature for the model', 0.0, 2.0, 1.0)
    st.session_state.max_tokens = st.slider(
        'Select the maximum tokens for the model', 1, 8000, 4000)

# -----------------------------------------------------------------------------------------------------------

# Step 1: Extract attributes from reviews and add keys to attributes


def extract_descriptive_details_from_reviews(reviews):
    model = st.session_state.model

    descriptive_details_list = []

    system_prompt = "Your aim for this task is to discard opinions and retain only descriptive information. For each review that you are provided you have to output the \"Opinions to be discarded\" and \"Extracted descriptive pairs\" which should be in the form of a list with key:value pairs."

    main_prompt = """input: For this task, I will give you some product reviews. Your job is to identify the following kinds of information in each review: 1. Descriptive: Describes specific attributes of the product, such as dimensions, size, color, materials, or specific functionalities.These are things that are objectively true about the product irrespective of the user. 2. Opinion-based: Includes subjective statements, personal experiences, and expressions of preference or feeling.\n\nHere are some ways you can identify opinions:\nAn opinion is a belief, judgment, or way of thinking about something that is subjective, often based on one's personal perspective, feelings, or tastes.  An \"opinion\" can include: Subjective Language and Qualitative Descriptions: Phrases that describe sensory qualities or characteristics based on personal experience or preference. This includes not only explicit statements of preference like \"I prefer\" or \"I like,\" but also any qualitative assessment of a product's features such as \"good bass,\" \"average loudness,\" or \"crisp display.\" These are subjective because they reflect individual perceptions and feelings about a sensory experience (like sound, taste, visual appeal), which can vary significantly from person to person.\nExpressions of Personal Experience or Preference: Statements that directly reflect individual experiences, preferences, or emotions. This includes not just explicit expressions like \"I love this feature,\" or \"I found the interface confusing,\" but also any description that implies a personal reaction or evaluation, such as \"the setup was easy\". Comparative and Relative Assessments: Judgments that compare the product or its features to others, or that use relative terms like \"good\", \"better,\" \"worse,\", \"great\", \"bad\", \"most,\" or \"least.\" These assessments are inherently subjective as they are based on personal criteria and perceptions. Lack of Objective or Verifiable Basis: Opinions are characterized by a lack of reliance on factual evidence and cannot be proven true or false in the same way factual statements can. They are more about personal interpretation and less about measurable, objective data. This includes any statement about a product that cannot be objectively measured or verified, like \"feels durable\" or \"looks elegant.\"  The attributes you should identify should be regarding the product the user is talking about and should not include any irrelevant information that might be present in the review provided by the user.\n\nYour first step would be to identify such opinions following the rules above and list them under the header of \"Opinions to be discarded\" and discard it from the review. After that, for each piece of descriptive information present in a review, return one or two-word attributes that describe the type of information. For example, if a review mentions 'satin', return the attribute 'fabric' or a synonym of fabric. Return the result under the header of \"Extracted descriptive pairs\" as a table with two columns - one for the attribute and another for the value or description. Create one table for each review. The response format must be a simple list of 'attribute: value' pairs, one per line.\nMake sure that opinions do not appear in this list. Anything listed under \"Opinions to be discarded\" should not appear in this section.\n\nReview: {review}"""

    review_examples = ["The speaker is very loud and has great bass. It is also lightweight and portable.",
                       "The fabric is made of satin and very soft. It can also be washed in a machine. I like it a lot.", "This is the best moisturizer I have used. It is not greasy at all and has a matte finish."]
    output_examples = ["Opinions to be discarded: \"great bass\"\nExtracted descriptive pairs\nvolume: loud\nweight: lightweight\nportability: portable",
                       "Opinions to be discarded: \"I like it a lot\"\nExtracted descriptive pairs\nfabric: satin\nmachine wash: allowed", "Opinions to be discarded: \"best moisturizer I have used\"\nExtracted descriptive pairs\ngreasy: no\nfinish: matte"]

    prompt_parts = [system_prompt]
    for review, output in zip(review_examples, output_examples):
        prompt_parts.append(main_prompt.format(review=review))
        prompt_parts.append(f"output: {output}")

    for review in reviews:
        model_final_prompt = prompt_parts.copy()
        model_final_prompt.append(main_prompt.format(review=review))
        model_final_prompt.append("output:")

        # generate a response and keep trying if you get internal server error
        response = None
        while response is None:
            try:
                response = model.generate_content(model_final_prompt)
            except Exception as e:
                if "internal error" in str(e).lower():
                    continue
                else:
                    raise e

        try:
            pairs = response.text.strip().split('\n')
        except:
            return "Safety Error!"

        review_dict = {}
        for pair in pairs:
            if ': ' in pair:
                key, value = pair.split(': ', 1)
                if "opinion" not in key.lower() and "extracted" not in key.lower():
                    review_dict[key.strip()] = value.strip()

        descriptive_details_list.append(review_dict)

    return descriptive_details_list


def clean_descriptive_details(descriptive_details):
    cleaned_descriptive_details = []
    for review in descriptive_details:
        cleaned_review = {}
        for key, value in review.items():
            cleaned_key = key.replace('-', '').lower().strip()
            cleaned_value = value.replace('-', '').lower().strip()
            cleaned_review[cleaned_key] = cleaned_value
        cleaned_descriptive_details.append(cleaned_review)
    return cleaned_descriptive_details

# -----------------------------------------------------------------------------------------------------------
# Step 2: Comparison with seller's description


def compare_with_seller_description(seller_description, reviews):
    model = st.session_state.model

    system_prompt = "For each review, the first line should be \"Output table for Review X\" where X is the review number. Following that,  return one table for each review with three columns - \"Attribute\", \"Value\", \"Description\". Each column should be separated by the | character. Do not use this character anywhere else.\nThe first column should have the attribute from the review, the second column should have the value, and the third column should mention whether the attribute-value pair is \"missing\" from the seller's description, \"matches\", \"partially matches\", or \"contradicts\" the seller's description, or \"expresses opinions\". If the attribute-value pair matches, partially matches, or contradicts the seller's description, provide supporting evidence from the seller's description in the \"Description\" column. The format is 'description >> evidence'. Strictly follow this format and ensure that you provide a valid evidence. The result should mandatorily be in the correct format."

    main_prompt = """input:Given attribute-value pairs for a product review, compare the attribute-value pairs with the seller-provided product description and find \n1. missing information - information present in the review but missing from the seller's description \n2. contradictory information - conflicting information in the review and the seller's description \n3. matching information - same information present in both the seller's description and the review \n4. partially matching information - information in the review slightly matches information in the seller's description but not completely \n5. opinion-based information - information in the review that does not give descriptive details rather comments on how good or bad an attribute of the product is - these are subjective pieces of information and might not be objectively true about the product.  \n\nAfter identifying the above information, return one table for each review with three columns - \"Attribute\", \"Value\", \"Description\". Each column should be separated by the | character. Do not use this character anywhere else. \nThe first column should have the attribute from the review, the second column should have the value, and the third column should mention whether the attribute-value pair is \"missing\" from the seller's description, \"matches\", \"partially matches\", or \"contradicts\" the seller's description, or \"expresses opinions\". \nIf the attribute-value pair matches, partially matches, or contradicts the seller's description, provide supporting evidence from the seller's description in the \"Description\" column. The format is 'description >> evidence'. Strictly follow this format and ensure that you provide a valid evidence.  \n\nFor example, if the attribute-value pair for a review is color: blue, but the seller's description says the color is red, the \"Description\" column should be \"contradiction >> color: red in seller's description\"\nIf the seller's description says the color is blue, the \"Description\" column should be \"matches >> color: blue in seller's description\" in the output table. \nIf the seller's description says the color is light blue, the \"Description\" column should be \"partially matches >> color: light blue in seller's description\" in the output table. \nIf the seller's description does not mention color, the \"Description\" column should have \"missing\". \nIf the attribute-value pair for the review says color: bright, the \"Description\" column should have \"expresses opinion\". \n\nMake sure to perform the above three steps for each review provided and return one well-formed table per review. \nDo this separately for each reivew and hence you should provide a separete table for each of the reviews.\nFor each table use the heading \"Output table for Review X\" where X is the review number.\n\nSeller Description: {seller_desc}\n\nAll reviews\n"""

    main_prompt = main_prompt.format(seller_desc=seller_description)

    for review in reviews:
        main_prompt += f"Review: {review}\n"

    prompt_parts = [system_prompt]
    prompt_parts.append(main_prompt)
    prompt_parts.append("output:")

    # generate a response and keep trying if you get internal server error
    response = None
    while response is None:
        try:
            response = model.generate_content(prompt_parts)
        except Exception as e:
            if "internal error" in str(e).lower():
                continue
            else:
                raise e

    try:
        return response.text.strip()
    except:
        return "Safety Error!"

# answering that the output given by the llm is in the correct format, if not, having to run the llm again - can't think of a better way to debug this really


def check_compare(review_tables):
    tables = review_tables.split("\n\n")
    for table in tables:
        rows = table.split("\n")
        for row in rows:
            if "Output table" in row or "Attribute" in row or "--" in row:
                continue
            row_data = row.split("|")
            if len(row_data) != 3:
                return False
    return True


def run_compare(seller_description, reviews):
    compare_result = False
    run_num = 0
    review_tables = ""

    while not compare_result and run_num < 1:
        review_tables = compare_with_seller_description(seller_description, reviews)
        compare_result = check_compare(review_tables)
        run_num += 1
    return review_tables


def pretty_print_review_tables(review_tables):
    tables = review_tables.split("\n\n")
    pretty_tables = []
    for table in tables:
        pretty_table = []
        rows = table.split("\n")
        for row in rows:
            try:
                if "Output table" in row or "Attribute" in row or "---" in row:
                    continue
                row_data = row.split("|")
                row_data = [data.strip() for data in row_data]
            except:
                continue

            if len(row_data) == 3:
                pretty_table.append(row_data)
        pretty_tables.append(pd.DataFrame(pretty_table, columns=[
                             "Attribute", "Value", "Description"]))
    return pretty_tables

# hihglighting the missing, contradictory, partially matching and opinion based information and change font color to black


def highlight(val):
    if "missing" in val:
        return f"background-color: #256b38"
    elif "contradicts" in val:
        return f"background-color: #730d2b"
    elif "partially matches" in val:
        return f"background-color: #280575"
    else:
        return ""
# -----------------------------------------------------------------------------------------------------------
# Step 3: Merge Tables (rule based method)


def merge_tables(review_tables):
    tables = review_tables.split("\n\n")
    all_merged_data = []
    error_log = []
    for i, table in enumerate(tables):
        rows = table.split("\n")
        for row in rows:
            try:
                if "Output table" in row or "Attribute" in row or "---" in row:
                    continue
                row = row.strip()
                if row == "":
                    continue
                row_data = row.split("|")
                row_data = [data.strip() for data in row_data]
                row_data.append(str(i))

                if ">>" in row_data[2]:
                    reason = row_data[2].split(">>")[1].strip()
                else:
                    reason = "no reason available"
                row_data.append(reason)

                if "match" in row_data[2] and "partially match" not in row_data[2]:
                    row_data[2] = "matches"
                elif "partially match" in row_data[2]:
                    row_data[2] = "partially matches"
                elif "contradict" in row_data[2]:
                    row_data[2] = "contradicts"
                elif "opinion" in row_data[2]:
                    row_data[2] = "opinion"
                elif "missing" in row_data[2]:
                    row_data[2] = "missing"
                else:
                    row_data[2] = "unknown"
            except Exception as e:
                error_log.append(f"Error in row {row_data}: {str(e)}")
                continue

            if len(row_data) == 5:
                all_merged_data.append(row_data)
            else:
                error_log.append(
                    f"Error in row {row_data}: Incorrect number of columns")
                continue

    merged_table = pd.DataFrame(all_merged_data, columns=[
                                "Attribute", "Value", "Description", "Review number", "Reason"])
    missing_info = merged_table[merged_table["Description"]
                                == "missing"].sort_values(by="Attribute")
    contradictory_info = merged_table[merged_table["Description"]
                                      == "contradicts"].sort_values(by="Attribute")
    partially_matching_info = merged_table[merged_table["Description"]
                                           == "partially matches"].sort_values(by="Attribute")

    missing_info = missing_info.drop(columns=["Description", "Reason"])
    contradictory_info = contradictory_info.drop(columns=['Description'])
    partially_matching_info = partially_matching_info.drop(columns=[
                                                           'Description'])

    missing_info = missing_info.dropna()
    contradictory_info = contradictory_info.dropna()
    partially_matching_info = partially_matching_info.dropna()

    missing_info = missing_info.groupby('Attribute').agg(
        {'Value': ', '.join, 'Review number': ', '.join}).reset_index()
    contradictory_info = contradictory_info.groupby('Attribute').agg(
        {'Value': ', '.join, 'Review number': ', '.join}).reset_index()
    partially_matching_info = partially_matching_info.groupby('Attribute').agg(
        {'Value': ', '.join, 'Review number': ', '.join}).reset_index()

    return missing_info, contradictory_info, partially_matching_info, error_log
# -----------------------------------------------------------------------------------------------------------S

# Step 4: Grouping attributes


def group_attributes(table):
    model = st.session_state.model

    system_prompt = "For this task, your output should be in the form of a dictionary which can be converted to a JSON consisting of key and value pairs. The dictionary should be under the heading \"DICTIONARY\". Additionally, at the end you should also provide a explanation behind your reasoning under the heading of \"EXPLANATION\". The result should mandatorily be in the correct format."

    main_prompt = """input:You will be given a list of attributes describing different features of a product. Your work is to group them into categories of related attributes and also give a name to the category. Try not to make classifications that are too fine grained as the aim is to generalize into categories for easier indexing. Your output should be in the format of dictionary where the key is the category name and the value is a list of attributes that belong to that category. The dictionary should be under the heading \"DICTIONARY\" and should strictly follow JSON formatting. Additionally, at the end you should also provide a explanation behind why you grouped the attributes into those categories under the heading of \"EXPLANATION\"\nList of attributes:\n{attributes}"""

    example_attributes = ["weight", "size",
                          "dimensions", "height", "color", "material"]
    example_output = "DICTIONARY\n{\"Physical Attributes\" : [\"weight\", \"size\", \"dimensions\", \"height\"], \"Material Attributes\" : [\"color\", \"material\"]\nEXPLANATION\nI grouped the attributes into 'Physical Attributes' because they all describe the physical characteristics of the product. Similarly, I grouped 'color' and 'material' into 'Material Attributes' because they both describe the material of the product"

    prompt_parts = [system_prompt]
    prompt_parts.append(main_prompt.format(
        attributes="\n".join(example_attributes)))
    prompt_parts.append(f"output: {example_output}")

    attributes = ""
    for index, row in table.iterrows():
        attributes += row['Attribute'] + '\n'

    # check if the list if empty - this is to minimize computation as well as hallucination from the model
    if not attributes:
        return {}

    model_final_prompt = prompt_parts.copy()
    model_final_prompt.append(main_prompt.format(attributes=attributes))
    model_final_prompt.append("output:")

    # generate a response and keep trying if you get internal server error
    response = None
    while response is None:
        try:
            response = model.generate_content(model_final_prompt)
        except Exception as e:
            if "internal error" in str(e).lower():
                continue
            else:
                raise e

    try:
        return response.text.strip()
    except:
        return "Safety Error!"
# -----------------------------------------------------------------------------------------------------------S
# Step 5: creating tables for the groups (rule based)


def extract_grouped_attributes(response):
    if not response:
        return {}, "", False
    
    errored = False
    try:
        response = response.split("DICTIONARY")[1].strip()
        dictionary = response.split("EXPLANATION")[0].strip()
        dictionary = json.loads(dictionary)
        explanation = response.split("EXPLANATION")[1].strip()
    except:
        dictionary = 'could not extract dictionary'
        explanation = 'could not extract explanation'
        errored = True

    return dictionary, explanation, errored


def split_tables(table, category_dict):
    for category, attributes in category_dict.items():
        for attribute in attributes:
            if attribute in table['Attribute'].values:
                table.loc[table['Attribute'] ==
                          attribute, 'Category'] = category

    # now split the table into those cateogries
    split_tables = {}
    for category in category_dict.keys():
        split_tables[category] = table[table['Category'] == category].drop(columns=[
                                                                           'Category'])

    return split_tables
# -----------------------------------------------------------------------------------------------------------S


def main():
    st.set_page_config(page_title='PRAISE', layout='wide')
    st.markdown('''# :red[PRAISE]: :red[P]roduct :red[R]eview :red[A]ttribute :red[I]nsight :red[S]tructuring :red[E]ngine''')
    st.markdown('## Initial Model Configuration')
    model_selection = st.selectbox('Select your preferred language model', [
                                   'Select model', 'Gemini 1.5 Pro', 'GPT-3.5', 'GPT-4', 'Llama 3.0 70B'])
    hyperparameters()

    if model_selection == 'Select model':
        st.write('Please select a model to proceed')
    else:
        if st.button('Configure model'):
            if model_selection == 'Gemini 1.5 Pro':
                model = gemini_setup(st.session_state.temp,
                                     st.session_state.max_tokens)
                st.session_state.model = model
                st.success('Model configured successfully')
            else:
                st.error(
                    'We currently on support Gemini 1.5 Pro, more models will be added soon.')
                
    st.divider()

    st.markdown('## Enter details of the product')

    st.markdown('### Enter the description of the product')
    with st.expander("Details"):
        st.write("The description of the product helps us know that the product listing states already, and helps us ground what additional information we can extract from the reviews. This is often the seller provided description that we can see in e-commerce websites. For example, here is a possible description for a laptop: This is a 13-inch laptop with a 4K display, 16GB RAM, and 512GB SSD storage. Please ensure that the description is in the form of a plain text")
    description = st.text_area("Enter description here", "")

    st.subheader("Enter the review(s) of the product")
    with st.expander("Details"):
        st.write("The reviews of the product is the list of comments given by users of the product. You can enter the review(s) in the text area below or upload a json file with the review(s). The reviews should be in the format of plain JSON only, where each entry of the review is a separate JSON object with the key 'review:'. Here are a few possible examples for how the reviews should be formatted. Please ensure that this formatting is strictly followed")
        st.caption("Example 1 (single review)")
        st.json([{"review": "The laptop is very fast and has a great display"}])
        st.caption("Example 2 (multiple reviews)")
        st.json([{"review": "The laptop is very fast and has a great display"}, {
                "review": "The battery life is not very good"}])
        st.write(
            "Please ensure that the same format is followed for review in both input or the file")
    review_text = st.text_area("Enter review(s) here", "")
    review_file = st.file_uploader("Alternatively you can upload a json file with the review(s)", type=['json'])

    if st.button('Submit'):
        if review_file:
            reviews = json.load(review_file)
            st.info("Obtained a json file with the review(s)")
        elif review_text:
            reviews = json.loads(review_text)
            st.info("Obtained the review(s) from the text area")
        else:
            st.write(
                "Please provide the review(s) in the text area or upload a json file")

        review = []
        for r in reviews:
            review.append(r['review'])
        
        with st.spinner('Processing the reviews'):
            st.divider()

            st.header('Results from the different steps of the pipeline')

            # ----------------------------------------------------------------------------------------------------
            # Extract descriptive details from the review
            descriptive_details = extract_descriptive_details_from_reviews(review)
            descriptive_details = clean_descriptive_details(descriptive_details)
            st.markdown('### :red[Step 1:] Descriptive details extracted from the review')
            st.markdown(
                '##### :gray[The extracted descriptive details from the review are shown below]')
            with st.expander('Details'):
                st.write('The descriptive details are the specific attributes of the product that are mentioned in the review. These attributes could be dimensions, size, color, materials, or specific functionalities of the product. It is important to ensure that we include attributes only about the product. The extracted details are shown below in the form of a table with the key and the attribute as columns.')
            st.write(descriptive_details)

            # ----------------------------------------------------------------------------------------------------
            # Compare with seller description
            reviews = descriptive_details
            review_tables = run_compare(description, reviews)
            st.markdown('### :red[Step 2:] Comparison with seller description')
            st.markdown(
                '##### :gray[The comparison of the extracted descriptive details from the review with the seller description is shown below]')
            with st.expander('Details'):
                st.write('The comparison table shows the missing information, contradictory information, matching information, partially matching information, and opinion-based information between the extracted details from the review and the seller description. The table shows the key, attribute, and the status of the information (missing, contradictory, matching, partially matching, or opinion).')
            pretty_review_tables = pretty_print_review_tables(review_tables)
            for i, pretty_table in enumerate(pretty_review_tables):
                st.markdown(f'#### :gray[Output table for Review {i+1}]')
                # add the highlight function to the table
                pretty_table = pretty_table.style.applymap(
                    highlight, subset=['Description'])
                st.dataframe(pretty_table)

            # ----------------------------------------------------------------------------------------------------
            # Merge tables
            missing_info, contradictory_info, partially_matching_info, error_log = merge_tables(
                review_tables)
            st.markdown('### :red[Step 3:] Merged tables as per category')
            st.markdown(
                '##### :gray[The merged table shows the missing and contradictory information across all reviews]')
            with st.expander('Details'):
                st.write('The merged table shows the missing and contradictory information across all reviews. The table has three columns - the first column contains the key from the review, the second column contains the attribute from the review, and the last column mentions the review number.')
            st.markdown('#### :gray[Missing information]')
            st.dataframe(missing_info)
            st.markdown('#### :gray[Contradictory information]')
            st.dataframe(contradictory_info)
            st.markdown('#### :gray[Partially matching information]')
            st.dataframe(partially_matching_info)

            # ----------------------------------------------------------------------------------------------------
            # group attributes
            missing_info_attr = group_attributes(missing_info)
            contradictory_info_attr = group_attributes(contradictory_info)
            partially_matching_info_attr = group_attributes(partially_matching_info)
            st.markdown('### :red[Step 4:] Group attributes into categories')
            st.markdown('##### :gray[The attributes from each table are grouped according to the categories they belong to and the explanation behind the grouping is provided]')
            with st.expander('Details'):
                st.write('The attributes from each table are grouped according to the categories they belong to and the explanation behind the grouping is provided. The output is in the form of a dictionary where the key is the category name and the value is a list of attributes that belong to that category. Additionally, the explanation behind why the attributes were grouped into those categories is also provided.')

            missing_info_dict, missing_info_explanation, missing_error = extract_grouped_attributes(
                missing_info_attr)
            st.markdown('#### :gray[Missing information Dictionary:]')
            st.write(missing_info_dict)
            st.markdown('#### :gray[Missing information Explanation:]')
            st.write(missing_info_explanation)

            contradictory_info_dict, contradictory_info_explanation, contra_error = extract_grouped_attributes(
                contradictory_info_attr)
            st.markdown('#### :gray[Contradictory information Dictionary:]')
            st.write(contradictory_info_dict)
            st.markdown('#### :gray[Contradictory information Explanation:]')
            st.write(contradictory_info_explanation)

            partially_matching_info_dict, partially_matching_info_explanation, partially_match_error = extract_grouped_attributes(
                partially_matching_info_attr)
            st.markdown('#### :gray[Partially matching information Dictionary:]')
            st.write(partially_matching_info_dict)
            st.markdown('#### :gray[Partially matching information Explanation:]')
            st.write(partially_matching_info_explanation)
            # ----------------------------------------------------------------------------------------------------
            # split tables
            st.subheader('Step 5: Split tables into categories')
            st.markdown(
                '##### The tables are split into categories based on the grouped attributes')
            with st.expander('Details'):
                st.write('The tables are split into categories based on the grouped attributes. The tables are split into different categories and the attributes are grouped accordingly.')
            st.markdown('#### Missing information')
            if not missing_error:
                missing_fin = split_tables(missing_info, missing_info_dict)
                for category, table in missing_fin.items():
                    st.markdown(f'##### :gray[{category}]')
                    st.dataframe(table)

            st.markdown('#### Contradictory information')
            if not contra_error:
                contradictory_fin = split_tables(
                    contradictory_info, contradictory_info_dict)
                for category, table in contradictory_fin.items():
                    st.markdown(f'##### :gray[{category}]')
                    st.dataframe(table)

            st.markdown('#### Partially matching information')
            if not partially_match_error:
                partially_matching_fin = split_tables(
                    partially_matching_info, partially_matching_info_dict)
                for category, table in partially_matching_fin.items():
                    st.markdown(f'##### :gray[{category}]')
                    st.dataframe(table)


if __name__ == '__main__':
    main()
