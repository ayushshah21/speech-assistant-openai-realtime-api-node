# Kayako AI Call Assistant Roadmap

This document outlines the remaining action items, requirements, and milestones for the Kayako AI Call Assistant project. It references the **existing** functionality (real-time conversation using OpenAI Realtime API + Twilio) and lays out the **additional** features and tasks needed for completion.

---

## 1. Current State

- **Real-Time AI Conversation**  
  We have integrated Twilio’s Media Streams with OpenAI’s Realtime API. The system can:
  - Receive and process live audio from a user calling a Twilio phone number.
  - Send AI-generated audio responses back to the user in real time.
  - Reference the **knowledge base** (KB) to answer questions about Kayako’s products and features.

- **Knowledge Base Context**  
  We have a `knowledge_base.json` that the AI consults for Kayako-specific questions.  
  Off-topic questions are (partially) handled by redirecting the user back to Kayako subjects.

---

## 2. Remaining Objectives

According to the **Product Requirements** and **User Stories**, we must add features that go beyond real-time QA. Below is a roadmap that details these requirements:

### 2.1. Automatic Ticket Creation in Kayako

**Goal**: When the AI ends a call (either because the question is answered or no KB match is found), the system automatically creates a ticket in Kayako with full call context.

**Tasks**:

1. **Collect user contact details**  
   - Capture user’s name, email, phone number, or relevant ID.  
   - Prompt user for missing info: “Could I have your email address to send you updates?”  
   - Store these details in a conversation state.

2. **Compile Transcript**  
   - Store the user’s utterances and the AI’s responses in a local transcript.  
   - On call end, generate a summary (or store the entire transcript verbatim).

3. **Use Kayako’s API** to create a new “Case” or “Ticket”  
   - Endpoints: Typically `/cases` with required fields (subject, description, requester, etc.).  
   - Attach the transcript and user details.  
   - If no KB match found, mark `requires_human_agent = true` and set priority to “Urgent.”

4. **Set up a test scenario**  
   - Confirm that once the call ends, the new ticket is visible in Kayako.  
   - Example: “Case #12345 created for user `<user@example.com>`” logs to console.

**Acceptance Criteria**:

- A Kayako ticket is automatically created after each call.  
- Ticket includes user details and call transcript.  
- If the user’s question was unanswered, set `requiresHumanFollowup`.

---

### 2.2. Escalation to Human Agent

**Goal**: If the AI can’t find an answer or the user explicitly wants a human, the call is ended politely and flagged for a human agent to follow up.

**Tasks**:

1. **Detect No KB Match**  
   - If the AI fails to find relevant info in the KB, it responds with a “We’ll connect you to a human.”
2. **End Call**  
   - Twilio logic to politely say: “A human will reach out soon. Thank you!”  
   - Hang up the call gracefully.
3. **Mark Ticket** as `ticket.requiresHumanAgent = true` in Kayako.  

**Acceptance Criteria**:

- AI politely informs the user that a human follow-up is needed.  
- The call is ended.  
- A ticket is generated with `requiresHumanFollowup` or appropriate high priority.

---

### 2.3. Additional KB Queries & Email Capture

**Goal**: The AI politely collects user email and reason for calling if not already provided.

**Tasks**:

1. **Conversation Manager**  
   - If the user’s email is unknown, ask: “May I have your email address for follow-up?”  
   - Store the email in a state.
2. **Flow**  
   - If user provides an email, confirm it.  
   - Continue providing answers from the KB or escalate if needed.  

**Acceptance Criteria**:

- AI can gather user email mid-call.  
- Email is stored for the final Kayako ticket.

---

### 2.4. Searching Knowledge Base by Email or Phone

**Goal**: If the user’s phone number or email is recognized in Kayako, the AI can check existing tickets or user info.

**Tasks**:

1. **(Optional) Kayako lookup**  
   - Use Kayako’s API to see if there’s an existing user/ticket for that phone/email.  
   - The AI can greet the user by name: “Hello, Jane, I see you’ve called about a password issue last time.”
2. **Integration**  
   - Add user ID to the newly created ticket so everything is tied to the correct user.

**Acceptance Criteria**:

- AI can identify returning callers or recognized emails.  
- The conversation references their existing Kayako profile.

---

### 2.5. Summaries & Sentiment Analysis (Optional Future)

**Goal**: Summarize calls or gauge user sentiment.

**Tasks** (Optional):

1. **AI Summaries**  
   - Summarize the conversation after call end to place in the Kayako ticket.
2. **Sentiment**  
   - Tag or add a note if the user seems frustrated or satisfied.

---

## 3. Architectural Overview

1. **Voice AI Engine**  
   - Already integrated: Twilio <-> OpenAI Realtime.  
   - Needs expansion to handle more advanced prompts + conversation flow states.
2. **Knowledge Base Integration**  
   - Already referencing `knowledge_base.json`.  
   - Must refine to handle “no match => escalate” logic.
3. **Call Handling**  
   - Already partially done.  
   - Must finalize ending calls, capturing user info, and creating tickets.
4. **Kayako Ticket Creation**  
   - Not yet fully implemented. Must connect using Kayako’s API endpoints to create or update a “Case.”

---

## 4. Detailed Action Plan

1. **Implement Ticket Creation**  
   - **Step**: Add final `onCallEnd` logic that calls Kayako’s `/cases` endpoint.  
   - **Who**: Developer assigned to Kayako API integration.  
   - **Time**: 1–2 days to test + confirm ticket fields.

2. **Collect User Email & Info**  
   - **Step**: Modify conversation flow to prompt for email if not known.  
   - **Who**: The same dev or a separate dev working on conversation states.  
   - **Time**: 1 day.

3. **Escalation**  
   - **Step**: If no answer found, politely end and set `requires_human_followup = true` in the Kayako ticket.  
   - **Who**: Conversation dev.  
   - **Time**: 0.5 day.

4. **(Optional) Lookup by Phone or Email**  
   - **Step**: If phone or email is recognized, greet user by name and fetch open tickets.  
   - **Who**: Kayako dev.  
   - **Time**: 2 days.

5. **Refine Real-Time Interaction**  
   - **Task**: Continue improving final word cutoff or barge-in experiences.  
   - **Time**: Ongoing.

6. **Final QA & Testing**  
   - **Task**: E2E test from call start to ticket creation.  
   - **Time**: 1–2 days.

---

## 5. Timeline & Milestones

**Week 1**  

- \[DONE\] Twilio <-> OpenAI Realtime integration for voice.  
- \[DONE\] Knowledge Base loading and basic referencing.

**Week 2**  

- \[IN PROGRESS\] Ticket creation in Kayako.  
- \[IN PROGRESS\] Collect user email + phone number.

**Week 3**  

- Escalation to a human if no KB match.  
- Basic flow for “off-topic => redirect back to Kayako.”

**Week 4**  

- Optional phone/email lookup in Kayako’s existing user records.  
- Final integration testing & bug fixes.

---

## 6. Final Notes

- **Daily Demos**: 11am CST. Slack channel is `#kayako-ai-dev`.  
- **Use Test Credentials**:  
  - Kayako test instance: [https://doug-test.kayako.com/agent/](https://doug-test.kayako.com/agent/)  
  - Creds: `anna.kim@trilogy.com / Kayakokayako1?`  
- **Focus** on the main use cases first (User Story 1–2–3–4), then handle advanced features if time allows.  
- **Ensure** you log calls and transcripts so we can quickly debug escalations or partial answers.

---

### Sample cURL Request to POST a New Conversation

```bash
curl -X "POST" "https://doug-test.kayako.com/api/v1/cases?include=channel,last_public_channel,mailbox,facebook_page,facebook_account,twitter_account,user,organization,sla_metric,sla_version_target,sla_version,identity_email,identity_domain,identity_facebook,identity_twitter,identity_phone,case_field,read_marker" \
     -H 'Content-Type: application/json; charset=UTF-8' \
     -u 'anna.kim@trilogy.com:Kayakokayako1?' \
     -d $'{
  "field_values": {
    "product": "80"
  },
  "status_id": "1",
  "attachment_file_ids": "",
  "tags": "gauntlet-ai",
  "type_id": 7,
  "channel": "MAIL",
  "subject": "[GAUNTLET AI TEST] Call Summary",
  "last_post_status": null,
  "last_replied_at": null,
  "brand_id": null,
  "latest_assignee_update": null,
  "contents": "<div>SUBJECT<br>&lt;subject goes here&gt;<br><br>SUMMARY:<br>&lt;2-3 line summary&gt;<br><br>PRIORITY ESTIMATE:<br>LOW<br><br>CALL TRANSCRIPT:<br>&lt;transcript&gt;</div>",
  "assigned_agent_id": "309",
  "read_marker_id": null,
  "_is_fully_loaded": false,
  "last_reply_by_requester_at": null,
  "form_id": "1",
  "last_updated_by_id": null,
  "assigned_team_id": "1",
  "pinned_notes_count": null,
  "requester_id": "309",
  "channel_id": "1",
  "last_message_preview": null,
  "priority_id": "1",
  "last_reply_by_agent_at": null,
  "channel_options": {
    "cc": [],
    "html": true
  },
  "last_public_channel_id": null
}'
