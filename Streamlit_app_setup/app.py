import streamlit as st


def main():

    st.set_page_config(
        page_title="Pyry's Streamlit testing", layout="wide"
    )
    st.title("Welcome to Pyry's Streamlit App!")

    st.markdown(
        """
        Hello!

        #### Technology

        * [Streamlit][streamlit] This user interface has been built using Streamlit module.
        
        #### Links

        * [Github][repository]

        [repository]: https://github.com/pyryjh/Portfolio
        [streamlit]: https://streamlit.io/

    """
    )

    st.markdown(
        """
        ---

        :exclamation: **Disclaimer: this application is under development and might not appear as expected.**

        """
    )
    st.markdown(
        """
        :x: **Disclaimer: Disclaimers here.**

        """
    )
    st.markdown(
        """
        	:bulb: **Ideas, feedback...**

        """
    )
    st.markdown(
        """
        	:memo: **Logging, maybe**

        """
    )


if __name__ == "__main__":
    
    main()