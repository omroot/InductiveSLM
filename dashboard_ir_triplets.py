"""
IR-Triplet Dataset Dashboard
Navigate and explore the Inductive Reasoning Triplet benchmark dataset
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from collections import Counter

# Page config
st.set_page_config(
    page_title="IR-Triplet Dataset Explorer",
    page_icon="🧠",
    layout="wide"
)

@st.cache_data
def load_dataset():
    """Load the IR-Triplet dataset"""
    data_path = Path("cache/raw_data/ir_triplets/ir_triplets.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

@st.cache_data
def get_statistics(data):
    """Calculate dataset statistics"""
    forms = [item['form'] for item in data]
    form_counts = Counter(forms)

    stats = {
        'total_triplets': len(data),
        'unique_forms': len(form_counts),
        'form_distribution': dict(form_counts),
        'avg_observation_length': sum(len(item['Training Observations']) for item in data) / len(data),
        'avg_question_length': sum(len(item['Question']) for item in data) / len(data),
        'avg_answer_length': sum(len(item['Answer']) for item in data) / len(data),
    }
    return stats

def main():
    st.title("🧠 IR-Triplet Dataset Explorer")
    st.markdown("""
    **Inductive Reasoning Triplet Benchmark** - Navigate and explore inductive reasoning tasks

    📄 [Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5529459) |
    💾 [GitHub](https://github.com/omroot/InductiveSLM)
    """)

    # Load data
    try:
        data = load_dataset()
        stats = get_statistics(data)
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure `cache/raw_data/ir_triplets/ir_triplets.json` exists.")
        return

    # Sidebar - Statistics
    with st.sidebar:
        st.header("📊 Dataset Statistics")
        st.metric("Total Triplets", stats['total_triplets'])
        st.metric("Unique Forms", stats['unique_forms'])

        st.markdown("---")
        st.subheader("Average Lengths")
        st.metric("Observation", f"{stats['avg_observation_length']:.0f} chars")
        st.metric("Question", f"{stats['avg_question_length']:.0f} chars")
        st.metric("Answer", f"{stats['avg_answer_length']:.0f} chars")

        st.markdown("---")
        st.subheader("Form Distribution")
        form_df = pd.DataFrame([
            {'Form': form, 'Count': count}
            for form, count in sorted(stats['form_distribution'].items(), key=lambda x: -x[1])
        ])
        st.dataframe(form_df, hide_index=True, use_container_width=True)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Browse", "📈 Statistics", "🎯 Filter", "📥 Export"])

    with tab1:
        st.header("Browse Triplets")

        # Navigation controls
        col1, col2, col3 = st.columns([2, 3, 2])

        with col1:
            page_size = st.selectbox("Items per page", [1, 5, 10, 25, 50], index=0)

        with col2:
            current_page = st.number_input(
                f"Page (1-{(len(data)-1)//page_size + 1})",
                min_value=1,
                max_value=(len(data)-1)//page_size + 1,
                value=1
            )

        with col3:
            jump_to = st.number_input(
                f"Jump to item (1-{len(data)})",
                min_value=1,
                max_value=len(data),
                value=1
            )
            if st.button("Go"):
                current_page = (jump_to - 1) // page_size + 1

        # Calculate indices
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(data))

        # Display items
        for idx in range(start_idx, end_idx):
            item = data[idx]

            with st.container():
                st.markdown(f"### Triplet #{idx + 1}")

                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.markdown("**Form:**")
                with col_b:
                    st.info(item['form'])

                st.markdown("**📝 Training Observations:**")
                st.text_area(
                    "Observations",
                    item['Training Observations'],
                    height=100,
                    key=f"obs_{idx}",
                    label_visibility="collapsed"
                )

                st.markdown("**❓ Question:**")
                st.text_area(
                    "Question",
                    item['Question'],
                    height=60,
                    key=f"q_{idx}",
                    label_visibility="collapsed"
                )

                st.markdown("**✅ Answer:**")
                st.text_area(
                    "Answer",
                    item['Answer'],
                    height=80,
                    key=f"a_{idx}",
                    label_visibility="collapsed"
                )

                st.markdown("---")

        # Pagination info
        st.caption(f"Showing items {start_idx + 1}-{end_idx} of {len(data)}")

    with tab2:
        st.header("Dataset Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Form Distribution")
            form_df = pd.DataFrame([
                {'Reasoning Form': form, 'Count': count, 'Percentage': f"{count/stats['total_triplets']*100:.1f}%"}
                for form, count in sorted(stats['form_distribution'].items(), key=lambda x: -x[1])
            ])
            st.dataframe(form_df, hide_index=True, use_container_width=True)

            # Bar chart
            st.bar_chart(
                form_df.set_index('Reasoning Form')['Count'],
                height=400
            )

        with col2:
            st.subheader("Length Statistics")
            length_df = pd.DataFrame([
                {'Field': 'Observation', 'Avg Length (chars)': int(stats['avg_observation_length'])},
                {'Field': 'Question', 'Avg Length (chars)': int(stats['avg_question_length'])},
                {'Field': 'Answer', 'Avg Length (chars)': int(stats['avg_answer_length'])},
            ])
            st.dataframe(length_df, hide_index=True, use_container_width=True)

            # Length distribution
            st.bar_chart(
                length_df.set_index('Field')['Avg Length (chars)'],
                height=400
            )

    with tab3:
        st.header("Filter by Reasoning Form")

        # Form filter
        all_forms = sorted(stats['form_distribution'].keys())
        selected_form = st.selectbox("Select reasoning form:", ["All"] + all_forms)

        if selected_form == "All":
            filtered_data = data
        else:
            filtered_data = [item for item in data if item['form'] == selected_form]

        st.info(f"Found {len(filtered_data)} triplets")

        # Display filtered results
        if filtered_data:
            display_limit = st.slider("Show first N results:", 1, min(50, len(filtered_data)), min(10, len(filtered_data)))

            for idx, item in enumerate(filtered_data[:display_limit]):
                with st.expander(f"Triplet {idx + 1}: {item['form']}"):
                    st.markdown("**📝 Observations:**")
                    st.write(item['Training Observations'])

                    st.markdown("**❓ Question:**")
                    st.write(item['Question'])

                    st.markdown("**✅ Answer:**")
                    st.write(item['Answer'])

            if len(filtered_data) > display_limit:
                st.caption(f"Showing {display_limit} of {len(filtered_data)} results. Use Export tab to get all.")

    with tab4:
        st.header("Export Data")

        # Form selection for export
        export_form = st.selectbox("Select form to export:", ["All"] + sorted(stats['form_distribution'].keys()))

        if export_form == "All":
            export_data = data
        else:
            export_data = [item for item in data if item['form'] == export_form]

        st.info(f"Selected {len(export_data)} triplets for export")

        # Export format
        export_format = st.radio("Export format:", ["JSON", "CSV", "Markdown"])

        if st.button("Generate Export"):
            if export_format == "JSON":
                export_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=export_str,
                    file_name=f"ir_triplets_{export_form.replace(' ', '_')}.json",
                    mime="application/json"
                )

            elif export_format == "CSV":
                export_df = pd.DataFrame(export_data)
                csv_str = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_str,
                    file_name=f"ir_triplets_{export_form.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

            elif export_format == "Markdown":
                md_lines = [f"# IR-Triplet Export: {export_form}\n"]
                for idx, item in enumerate(export_data):
                    md_lines.append(f"\n## Triplet {idx + 1}\n")
                    md_lines.append(f"**Form:** {item['form']}\n")
                    md_lines.append(f"\n**Observations:**\n{item['Training Observations']}\n")
                    md_lines.append(f"\n**Question:**\n{item['Question']}\n")
                    md_lines.append(f"\n**Answer:**\n{item['Answer']}\n")
                    md_lines.append("\n---\n")

                md_str = "\n".join(md_lines)
                st.download_button(
                    label="Download Markdown",
                    data=md_str,
                    file_name=f"ir_triplets_{export_form.replace(' ', '_')}.md",
                    mime="text/markdown"
                )

            st.success("Export ready for download!")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Citation:**
    ```bibtex
    @article{missaoui2025inductive,
      title={Inductive Triplet Fine-Tuning for Small Language Models},
      author={Missaoui, Oualid},
      journal={SSRN Electronic Journal},
      year={2025},
      doi={10.2139/ssrn.5529459}
    }
    ```
    """)

if __name__ == "__main__":
    main()
