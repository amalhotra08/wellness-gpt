"""add participant pin to users

Revision ID: 20260429_add_participant_pin
Revises: 
Create Date: 2026-04-29

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260429_add_participant_pin"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("participant_pin", sa.String(length=12), nullable=True))
    op.create_index("ix_users_participant_pin", "users", ["participant_pin"], unique=True)


def downgrade():
    op.drop_index("ix_users_participant_pin", table_name="users")
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("participant_pin")
