"""Data models for Signal messages and conversations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Conversation:
    id: str
    type: str  # 'private' or 'group'
    name: str | None = None
    profile_name: str | None = None
    profile_full_name: str | None = None
    phone: str | None = None
    service_id: str | None = None
    group_id: str | None = None
    active_at: int | None = None
    member_count: int | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.profile_full_name or self.profile_name or self.phone or self.id[:8]

    @property
    def active_date(self) -> datetime | None:
        if self.active_at:
            return datetime.fromtimestamp(self.active_at / 1000, tz=timezone.utc)
        return None


@dataclass
class Message:
    id: int
    body: str
    conversation_id: str
    type: str  # 'incoming' or 'outgoing'
    sent_at: int  # unix timestamp in milliseconds
    source_service_id: str | None = None
    has_attachments: bool = False
    expire_timer: int | None = None
    conversation_name: str | None = None
    sender_name: str | None = None
    json_data: dict = field(default_factory=dict, repr=False)

    @property
    def sent_date(self) -> datetime:
        return datetime.fromtimestamp(self.sent_at / 1000, tz=timezone.utc)

    @property
    def is_outgoing(self) -> bool:
        return self.type == "outgoing"
