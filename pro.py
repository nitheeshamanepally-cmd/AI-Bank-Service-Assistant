import streamlit as st
import chromadb
import ollama

from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Bank Service Assistant",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# BLACK BANKING UI
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #000000;
    }

    [data-testid="stHeader"] {
        background-color: #000000;
    }

    [data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #222222;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    p, label {
        color: #ffffff !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #080808;
        border: 1px solid #222222;
        border-radius: 10px;
    }

    [data-testid="stChatMessage"] p {
        color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #0b0b0b !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #777777 !important;
    }

    .stButton > button {
        background-color: #0b0b0b !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }

    .stButton > button:hover {
        background-color: #111111 !important;
        color: #00bfff !important;
        border: 1px solid #00bfff !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL SETTINGS
# =========================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_MODEL = "gemma3:1b"


# =========================================================
# BANK KNOWLEDGE BASE
# =========================================================
#
# NO TXT FILES
# NO PDF
# NO UPLOAD
#
# All information is stored directly in Python.
# =========================================================

BANK_DATA = {

    "SBI": {

        "full_name": "State Bank of India",

        "information": [

            # -------------------------------------------------
            # BANK
            # -------------------------------------------------

            """
            State Bank of India, commonly called SBI, is a major
            public sector bank in India. It provides banking,
            financial and digital banking services.
            """,

            # -------------------------------------------------
            # SAVINGS ACCOUNT
            # -------------------------------------------------

            """
            SBI savings accounts are used for regular banking
            activities such as deposits, withdrawals, transfers
            and digital payments.
            """,

            """
            To open a savings account, customers may generally
            need identity proof, address proof, PAN or applicable
            tax information, photographs and KYC verification.
            Exact requirements depend on the account type.
            """,

            # -------------------------------------------------
            # CURRENT ACCOUNT
            # -------------------------------------------------

            """
            SBI current accounts are generally used by businesses
            and customers who require frequent transactions.
            """,

            # -------------------------------------------------
            # FIXED DEPOSIT
            # -------------------------------------------------

            """
            SBI fixed deposits allow customers to deposit money
            for a selected period at applicable interest terms.
            The applicable interest rate and maturity amount
            depend on the deposit scheme and tenure.
            """,

            # -------------------------------------------------
            # RECURRING DEPOSIT
            # -------------------------------------------------

            """
            SBI recurring deposits allow customers to make
            regular deposits for a selected period according
            to the applicable scheme.
            """,

            # -------------------------------------------------
            # HOME LOAN
            # -------------------------------------------------

            """
            SBI provides home loan facilities for eligible
            applicants subject to applicable eligibility,
            income, credit and property requirements.
            """,

            """
            Common documents that may be required for a home
            loan application include identity proof, address
            proof, PAN, photographs, income proof, salary slips
            for salaried applicants, Form 16 or income tax
            documents, recent bank statements, employment or
            business proof and property-related documents.
            """,

            """
            Property-related home loan documents may include
            sale agreement, title documents, property papers,
            approved building or property documents and other
            documents requested during verification.
            """,

            """
            The exact home loan documents depend on the applicant,
            employment type, property, loan type and bank
            requirements. Applicants should verify the latest
            requirements with the bank.
            """,

            # -------------------------------------------------
            # PERSONAL LOAN
            # -------------------------------------------------

            """
            SBI personal loans are subject to applicable
            eligibility conditions and verification.
            Common documents may include identity proof,
            address proof, income proof, salary slips and
            bank statements depending on the applicant.
            """,

            # -------------------------------------------------
            # EDUCATION LOAN
            # -------------------------------------------------

            """
            SBI education loans are available for eligible
            students subject to applicable conditions.
            Documents may include student identity documents,
            admission or course documents, academic records,
            fee details and income or KYC documents.
            """,

            # -------------------------------------------------
            # VEHICLE LOAN
            # -------------------------------------------------

            """
            SBI provides vehicle loan facilities for eligible
            customers subject to applicable conditions,
            documentation and vehicle-related requirements.
            """,

            # -------------------------------------------------
            # DEBIT CARD
            # -------------------------------------------------

            """
            SBI provides debit card facilities to eligible
            account holders. Debit cards can be used for
            supported ATM, POS and online transactions.
            """,

            # -------------------------------------------------
            # CREDIT CARD
            # -------------------------------------------------

            """
            SBI provides credit card products subject to
            product availability, eligibility and approval.
            """,

            # -------------------------------------------------
            # ATM
            # -------------------------------------------------

            """
            SBI ATM services may support cash withdrawal,
            balance enquiry, PIN related services and other
            supported ATM operations.
            """,

            # -------------------------------------------------
            # UPI
            # -------------------------------------------------

            """
            SBI supports UPI based digital payment services
            through supported banking applications and channels.
            """,

            # -------------------------------------------------
            # MOBILE BANKING
            # -------------------------------------------------

            """
            SBI provides mobile banking facilities through
            supported digital banking applications and services.
            """,

            # -------------------------------------------------
            # INTERNET BANKING
            # -------------------------------------------------

            """
            SBI provides internet banking facilities for
            supported banking services such as account access,
            transfers and statements.
            """,

            # -------------------------------------------------
            # NEFT
            # -------------------------------------------------

            """
            NEFT is an electronic fund transfer system that
            can be used to transfer money between supported
            bank accounts through applicable banking channels.
            """,

            # -------------------------------------------------
            # RTGS
            # -------------------------------------------------

            """
            RTGS is an electronic fund transfer system used
            for supported bank-to-bank fund transfers.
            """,

            # -------------------------------------------------
            # IMPS
            # -------------------------------------------------

            """
            IMPS is an electronic fund transfer service that
            supports fast fund transfers through applicable
            banking channels.
            """,

            # -------------------------------------------------
            # KYC
            # -------------------------------------------------

            """
            KYC means Know Your Customer. Banks use KYC to
            verify customer identity and address according
            to applicable requirements.
            """,

            # -------------------------------------------------
            # STATEMENTS
            # -------------------------------------------------

            """
            SBI customers can obtain account statements through
            supported banking channels.
            """,

            # -------------------------------------------------
            # SECURITY
            # -------------------------------------------------

            """
            Customers should never share OTP, ATM PIN, UPI PIN,
            CVV, passwords, card details or internet banking
            credentials with anyone.
            """
        ]
    },


    # =========================================================
    # IOB
    # =========================================================

    "IOB": {

        "full_name": "Indian Overseas Bank",

        "information": [

            """
            Indian Overseas Bank, commonly called IOB, is a
            public sector bank in India providing banking and
            financial services.
            """,

            """
            IOB savings accounts are used for regular banking
            activities such as deposits, withdrawals, transfers
            and digital payments.
            """,

            """
            To open a savings account, customers may generally
            need identity proof, address proof, PAN or applicable
            tax information, photographs and KYC verification.
            Exact requirements depend on the account type.
            """,

            """
            IOB current accounts are generally used by businesses
            and customers requiring frequent transactions.
            """,

            """
            IOB fixed deposits allow customers to deposit money
            for a selected period according to applicable terms.
            """,

            """
            IOB recurring deposits allow customers to make
            regular deposits for a selected period.
            """,

            """
            IOB provides home loan facilities for eligible
            applicants subject to applicable eligibility,
            income, credit and property requirements.
            """,

            """
            Common documents that may be required for an IOB
            home loan include identity proof, address proof,
            PAN, photographs, income proof, salary slips,
            Form 16 or income tax documents, bank statements,
            employment or business proof and property-related
            documents.
            """,

            """
            Property-related documents for a home loan may include
            sale agreement, title documents, property papers,
            approved property documents and other documents
            requested during verification.
            """,

            """
            Exact home loan documentation depends on the applicant,
            employment type, property, loan type and bank requirements.
            Customers should verify the latest requirements with IOB.
            """,

            """
            IOB provides personal loan facilities for eligible
            customers subject to applicable conditions and
            documentation requirements.
            """,

            """
            Personal loan documents may generally include identity
            proof, address proof, income proof, salary slips and
            bank statements depending on the applicant.
            """,

            """
            IOB provides education loan services for eligible
            students subject to applicable requirements.
            """,

            """
            Education loan documentation may include student
            identity documents, admission documents, academic
            records, fee details and applicable KYC or income
            documents.
            """,

            """
            IOB provides vehicle loan facilities for eligible
            applicants subject to applicable conditions.
            """,

            """
            IOB provides debit card facilities to eligible
            account holders.
            """,

            """
            IOB provides applicable credit card or card-related
            services subject to product availability and eligibility.
            """,

            """
            IOB ATM services support applicable ATM operations
            such as cash withdrawal and balance enquiry.
            """,

            """
            IOB supports UPI based digital payment services
            through supported banking channels.
            """,

            """
            IOB provides mobile banking facilities through
            supported digital banking channels.
            """,

            """
            IOB provides internet banking facilities for
            supported banking services.
            """,

            """
            IOB supports NEFT electronic fund transfers through
            applicable banking channels.
            """,

            """
            IOB supports RTGS electronic fund transfers through
            applicable banking channels.
            """,

            """
            IOB supports IMPS electronic fund transfers through
            applicable banking channels.
            """,

            """
            KYC means Know Your Customer. IOB uses KYC procedures
            to verify customer identity and address according to
            applicable requirements.
            """,

            """
            IOB customers can obtain account statements through
            supported banking channels.
            """,

            """
            Customers should never share OTP, ATM PIN, UPI PIN,
            CVV, passwords, card details or internet banking
            credentials with anyone.
            """
        ]
    },


    # =========================================================
    # CANARA BANK
    # =========================================================

    "Canara Bank": {

        "full_name": "Canara Bank",

        "information": [

            """
            Canara Bank is a major public sector bank in India
            providing banking and financial services.
            """,

            """
            Canara Bank savings accounts are used for regular
            banking activities such as deposits, withdrawals,
            transfers and digital payments.
            """,

            """
            To open a savings account, customers may generally
            need identity proof, address proof, PAN or applicable
            tax information, photographs and KYC verification.
            Exact requirements depend on the account type.
            """,

            """
            Canara Bank current accounts are generally used by
            businesses and customers requiring frequent transactions.
            """,

            """
            Canara Bank provides fixed deposit facilities for
            customers who want to deposit money for a selected
            period according to applicable terms.
            """,

            """
            Canara Bank provides recurring deposit facilities
            for customers making regular deposits.
            """,

            """
            Canara Bank provides home loan facilities for eligible
            applicants subject to applicable eligibility, income,
            credit and property requirements.
            """,

            """
            Common documents that may be required for a Canara
            Bank home loan include identity proof, address proof,
            PAN, photographs, income proof, salary slips, Form 16
            or income tax documents, bank statements, employment
            or business proof and property-related documents.
            """,

            """
            Property-related home loan documents may include
            sale agreement, title documents, property papers,
            approved property documents and other documents
            requested during verification.
            """,

            """
            Exact home loan documentation depends on the applicant,
            employment type, property, loan type and bank requirements.
            Customers should verify the latest requirements with
            Canara Bank.
            """,

            """
            Canara Bank provides personal loan facilities subject
            to applicable eligibility and documentation requirements.
            """,

            """
            Personal loan documents may generally include identity
            proof, address proof, income proof, salary slips and
            recent bank statements depending on the applicant.
            """,

            """
            Canara Bank provides education loan facilities for
            eligible students subject to applicable conditions.
            """,

            """
            Education loan documentation may include student identity
            documents, admission documents, academic records,
            fee details and applicable KYC or income documents.
            """,

            """
            Canara Bank provides vehicle loan facilities subject
            to applicable eligibility and documentation requirements.
            """,

            """
            Canara Bank provides debit card services to eligible
            account holders.
            """,

            """
            Canara Bank provides applicable credit card services
            subject to product availability and eligibility.
            """,

            """
            Canara Bank ATM services support applicable operations
            such as cash withdrawal and balance enquiry.
            """,

            """
            Canara Bank supports UPI based digital payment services
            through supported banking channels.
            """,

            """
            Canara Bank provides mobile banking facilities through
            supported digital banking channels.
            """,

            """
            Canara Bank provides internet banking facilities for
            supported banking services.
            """,

            """
            Canara Bank supports NEFT electronic fund transfers
            through applicable banking channels.
            """,

            """
            Canara Bank supports RTGS electronic fund transfers
            through applicable banking channels.
            """,

            """
            Canara Bank supports IMPS electronic fund transfers
            through applicable banking channels.
            """,

            """
            KYC means Know Your Customer. Canara Bank uses KYC
            procedures to verify customer identity and address
            according to applicable requirements.
            """,

            """
            Canara Bank customers can obtain account statements
            through supported banking channels.
            """,

            """
            Customers should never share OTP, ATM PIN, UPI PIN,
            CVV, passwords, card details or internet banking
            credentials with anyone.
            """
        ]
    }
}


# =========================================================
# SESSION STATE
# =========================================================

if "selected_bank" not in st.session_state:
    st.session_state.selected_bank = "SBI"


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "current_chat" not in st.session_state:
    st.session_state.current_chat = []


if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None


# =========================================================
# LOAD SENTENCE TRANSFORMER
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


embedding_model = load_embedding_model()


# =========================================================
# LOAD CHROMADB
# =========================================================

@st.cache_resource
def load_database():

    client = chromadb.PersistentClient(
        path="bank_db"
    )

    collection = client.get_or_create_collection(
        name="bank_rag_v3"
    )

    return collection


collection = load_database()


# =========================================================
# CREATE RAG DATABASE
# =========================================================

def create_rag_database():

    if collection.count() > 0:
        return

    documents = []
    ids = []
    metadatas = []

    for bank_name, bank_details in BANK_DATA.items():

        information = bank_details["information"]

        for index, text in enumerate(information):

            documents.append(
                text.strip()
            )

            ids.append(
                f"{bank_name}_{index}"
            )

            metadatas.append(
                {
                    "bank": bank_name
                }
            )

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


create_rag_database()


# =========================================================
# TITLE
# =========================================================

st.title(
    " AI Bank Service Assistant!🏦"
)




# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🏦 Bank Selection")

    selected_bank = st.selectbox(
        "Choose Bank",
        [
            "SBI",
            "IOB",
            "Canara Bank"
        ],
        index=[
            "SBI",
            "IOB",
            "Canara Bank"
        ].index(
            st.session_state.selected_bank
        )
    )


    # -----------------------------------------------------
    # BANK CHANGE
    # -----------------------------------------------------

    if selected_bank != st.session_state.selected_bank:

        st.session_state.selected_bank = selected_bank

        st.session_state.current_chat = []

        st.session_state.selected_chat = None

        st.rerun()


    st.divider()


    # =====================================================
    # CHAT HISTORY
    # =====================================================

    st.subheader("💬 Chat History")


    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.current_chat = []

        st.session_state.selected_chat = None

        st.rerun()


    if st.session_state.chat_history:

        for index, chat in enumerate(
            st.session_state.chat_history
        ):

            title = chat["title"]

            if st.button(
                f"💬 {title}",
                key=f"history_{index}",
                use_container_width=True
            ):

                st.session_state.current_chat = (
                    chat["messages"].copy()
                )

                st.session_state.selected_chat = index

                st.session_state.selected_bank = (
                    chat["bank"]
                )

                st.rerun()

    else:

        st.caption(
            "No previous chats"
        )


    st.divider()


    # =====================================================
    # CLEAR HISTORY
    # =====================================================

    if st.button(
        "🗑️ Clear All History",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.session_state.current_chat = []

        st.session_state.selected_chat = None

        st.rerun()


    st.divider()


    # =====================================================
    # BANKING SERVICES
    # =====================================================

    st.subheader("Banking Services")

    st.write("💰 Accounts")

    st.write("💵 Deposits")

    st.write("🏠 Loans")

    st.write("💳 Cards")

    st.write("🏧 ATM")

    st.write("📱 Mobile Banking")

    st.write("🌐 Internet Banking")

    st.write("💸 UPI & Payments")

    st.write("🧾 KYC")

    st.write("📄 Statements")

    st.write("🔐 Security")


# =========================================================
# SELECTED BANK
# =========================================================

selected_bank = st.session_state.selected_bank

st.info(
    f"🏦 Selected Bank: {selected_bank} — "
    f"{BANK_DATA[selected_bank]['full_name']}"
)


# =========================================================
# SUPPORTED BANKS
# =========================================================

st.header("Supported Banks")


bank1, bank2, bank3 = st.columns(3)


# =========================================================
# SBI
# =========================================================

with bank1:

    st.subheader("🏦 SBI")

    st.write(
        "State Bank of India"
    )


    if st.button(
        "Select SBI",
        key="sbi_button",
        use_container_width=True
    ):

        st.session_state.selected_bank = "SBI"

        st.session_state.current_chat = []

        st.session_state.selected_chat = None

        st.rerun()


# =========================================================
# IOB
# =========================================================

with bank2:

    st.subheader("🏦 IOB")

    st.write(
        "Indian Overseas Bank"
    )

    

    if st.button(
        "Select IOB",
        key="iob_button",
        use_container_width=True
    ):

        st.session_state.selected_bank = "IOB"

        st.session_state.current_chat = []

        st.session_state.selected_chat = None

        st.rerun()


# =========================================================
# CANARA
# =========================================================

with bank3:

    st.subheader("🏦 Canara Bank")

    st.write(
        "Canara Bank"
    )


    if st.button(
        "Select Canara",
        key="canara_button",
        use_container_width=True
    ):

        st.session_state.selected_bank = "Canara Bank"

        st.session_state.current_chat = []

        st.session_state.selected_chat = None

        st.rerun()


# =========================================================
# BANKING SERVICES
# =========================================================

st.header("Banking Services")


service1, service2, service3, service4 = st.columns(4)


with service1:

    st.write("💰 **Accounts**")


with service2:

    st.write("💵 **Deposits**")


with service3:

    st.write("🏠 **Loans**")


with service4:

    st.write("💳 **Cards**")


service5, service6, service7, service8 = st.columns(4)


with service5:

    st.write("🏧 **ATM**")


with service6:

    st.write("📱 **Mobile Banking**")


with service7:

    st.write("🌐 **Internet Banking**")


with service8:

    st.write("💸 **UPI & Payments**")


# =========================================================
# QUICK QUESTIONS
# =========================================================

st.header("Quick Questions")


q1, q2, q3 = st.columns(3)


with q1:

    if st.button(
        "🏠 Home Loan Documents",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "What documents are required to apply for a home loan?"
        )


with q2:

    if st.button(
        "💳 Debit Card",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "Tell me about debit card services."
        )


with q3:

    if st.button(
        "📱 Mobile Banking",
        use_container_width=True
    ):

        st.session_state.quick_question = (
            "What mobile banking services are available?"
        )


# =========================================================
# DISPLAY CURRENT CHAT
# =========================================================

for message in st.session_state.current_chat:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    f"Ask about {selected_bank} banking services..."
)


# =========================================================
# QUICK QUESTION INPUT
# =========================================================

if "quick_question" in st.session_state:

    question = st.session_state.quick_question

    del st.session_state.quick_question


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.current_chat.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.write(question)


    # =====================================================
    # CREATE QUESTION EMBEDDING
    # =====================================================

    query_embedding = embedding_model.encode(
        question
    ).tolist()


    # =====================================================
    # SEARCH CHROMADB
    # =====================================================

    results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=4,

        where={
            "bank": selected_bank
        }
    )


    # =====================================================
    # GET RETRIEVED INFORMATION
    # =====================================================

    documents = results.get(
        "documents",
        [[]]
    )


    if documents and documents[0]:

        context = "\n\n".join(
            documents[0]
        )

    else:

        context = (
            "No relevant bank information was found."
        )


    # =====================================================
    # RAG PROMPT
    # =====================================================

    prompt = f"""

You are an AI Bank Service Assistant.

Selected Bank:
{selected_bank}

Your task is to answer the user's banking question
using the retrieved information below.

IMPORTANT RULES:

1. Answer only using the retrieved information.
2. Do not invent bank policies.
3. Do not mix information from different banks.
4. Keep the answer simple and clear.
5. If exact information is not available, say:
   "I don't have that information in my current bank knowledge."
6. This is an educational banking assistant.
7. Do not claim that you can perform real transactions.

SECURITY RULE:

Never ask the user for:

- OTP
- ATM PIN
- UPI PIN
- CVV
- Password
- Card number
- Internet banking credentials

Retrieved Information:

{context}

User Question:

{question}

Answer clearly and briefly.
"""


    # =====================================================
    # OLLAMA
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching bank information..."
        ):

            try:

                response = ollama.chat(

                    model=OLLAMA_MODEL,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                answer = response[
                    "message"
                ][
                    "content"
                ]

            except Exception as error:

                answer = (
                    "⚠️ Ollama connection error.\n\n"
                    "Please make sure Ollama is running "
                    "and Gemma 3:1b is installed.\n\n"
                    f"Error: {error}"
                )


        st.write(answer)


    # =====================================================
    # SAVE AI MESSAGE
    # =====================================================

    st.session_state.current_chat.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # =====================================================
    # SAVE / UPDATE CHAT HISTORY
    # =====================================================

    if st.session_state.selected_chat is None:

        st.session_state.chat_history.append(
            {
                "bank": selected_bank,
                "title": question[:35],
                "messages": (
                    st.session_state.current_chat.copy()
                )
            }
        )

        st.session_state.selected_chat = (
            len(st.session_state.chat_history) - 1
        )

    else:

        st.session_state.chat_history[
            st.session_state.selected_chat
        ]["messages"] = (
            st.session_state.current_chat.copy()
        )


    # =====================================================
    # RERUN TO UPDATE SIDEBAR HISTORY
    # =====================================================

    st.rerun()


