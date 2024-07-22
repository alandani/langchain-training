from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from modules.linkedin import scrape_linkedin_profile
from modules.agents import linkedin_lookup_agent
from modules.output_parser import summary_parser

from dotenv import load_dotenv

def ice_break_with(name: str) -> str:
    linkedin_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(url=linkedin_url, mock=True)

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    \n{format_instruction}
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        partial_variables={"format_instruction": summary_parser.get_format_instructions()},
        template=summary_template,
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = summary_prompt_template | llm | summary_parser

    res = chain.invoke(input={"information": linkedin_data})

    print(res)


if __name__ == "__main__":
    load_dotenv()

    print("Chat Profile")
    ice_break_with(name="Eden Marco Udemy")